use std::collections::HashMap;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::io::{self, BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use fnv::FnvBuildHasher;
use log::{info, warn};
use structopt::StructOpt;

#[derive(Debug)]
pub struct Stats {
    pub year: i32,
    pub total_count: u32,
    pub book_count: u32,
}

#[derive(StructOpt)]
pub struct Opt {
    /// First year to consider from corpus.
    #[structopt(long)]
    pub from: i32,

    /// Last year to consider from corpus (inclusive).
    #[structopt(long)]
    pub to: i32,

    /// Input directory
    pub input_dir: PathBuf,
}

pub struct NGramFile {
    reader: BufReader<GzDecoder<fs::File>>,
    buf: String,
    first_year: i32,
    last_year: i32,
}

pub enum Event<W> {
    NGram(W, Stats),
    Skip,
    EndOfFile,
}

impl NGramFile {
    pub fn open(path: impl AsRef<Path>, first_year: i32, last_year: i32) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(GzDecoder::new(file));
        let buf = String::new();
        Ok(Self {
            reader,
            buf,
            first_year,
            last_year,
        })
    }

    #[inline]
    pub fn next<'a, W>(&'a mut self, template: W) -> io::Result<Event<W>>
    where
        W: Copy + AsMut<[&'a str]>,
    {
        self.buf.clear();
        if self.reader.read_line(&mut self.buf)? == 0 {
            return Ok(Event::EndOfFile);
        }
        let mut suffix = self.buf.as_str();

        let mut words = template;
        for word in words.as_mut().iter_mut() {
            if let Some((parsed_word, new_suffix)) = read_word(suffix) {
                *word = parsed_word;
                suffix = new_suffix;
            } else {
                return Ok(Event::Skip);
            }
        }

        let (year, new_suffix) = read_int(suffix, b'\t');
        suffix = new_suffix;
        let (total_count, new_suffix) = read_int(suffix, b'\t');
        suffix = new_suffix;
        let (book_count, new_suffix) = read_int(suffix, b'\n');
        assert!(new_suffix.is_empty());

        if year >= self.first_year && year <= self.last_year {
            Ok(Event::NGram(
                words,
                Stats {
                    year,
                    total_count,
                    book_count,
                },
            ))
        } else {
            Ok(Event::Skip)
        }
    }
}

pub fn iterate_ngram_files(opt: &Opt) -> io::Result<impl Iterator<Item = io::Result<NGramFile>>> {
    assert!(opt.input_dir.is_dir());
    let (first_year, last_year) = (opt.from, opt.to);

    Ok(
        fs::read_dir(&opt.input_dir)?.filter_map(move |entry| match entry {
            Ok(entry) => {
                let path = entry.path();
                if path.extension().map(OsStr::to_str) != Some(Some("gz")) {
                    warn!(
                        "Skipped file {}.",
                        path.file_name().unwrap().to_str().unwrap()
                    );
                    None
                } else {
                    info!(
                        "Reading file {} ...",
                        path.file_name().unwrap().to_str().unwrap()
                    );
                    Some(NGramFile::open(path, first_year, last_year))
                }
            }
            Err(e) => Some(Err(e)),
        }),
    )
}

#[inline(always)]
fn read_word(source: &str) -> Option<(&str, &str)> {
    unsafe {
        let bytes = source.as_bytes();
        assert!(bytes.len() >= 2);
        assert_ne!(*bytes.get_unchecked(0), b'\t');
        // Offset by one to allow underscore at first byte.
        let mut len = 1;
        for b in bytes.get_unchecked(1..) {
            match b {
                b'\t' | b' ' => break,
                b'_' => return None,
                _ => len += 1,
            }
        }

        if len == 1 && *bytes.get_unchecked(0) == b' ' {
            // Space is not a word, but it appears quite often in the corpus
            // (probably for annotations).
            return None;
        }
        let word = std::str::from_utf8_unchecked(&bytes.get_unchecked(..len));
        let suffix = std::str::from_utf8_unchecked(&bytes.get_unchecked(len + 1..));
        Some((word, suffix))
    }
}

#[inline]
fn read_int<T>(source: &str, next_byte: u8) -> (T, &str)
where
    T: From<u8> + std::ops::AddAssign + std::ops::MulAssign,
{
    unsafe {
        assert!(next_byte < 128);
        let bytes = source.as_bytes();
        assert!(!bytes.is_empty());
        let mut result = T::from(bytes.get_unchecked(0) - b'0');
        let mut skip = 2;

        for b in bytes.get_unchecked(1..) {
            match b {
                b'0'..=b'9' => {
                    result *= T::from(10);
                    result += T::from(b - b'0');
                    skip += 1;
                }
                &b if b == next_byte => break,
                _ => panic!(),
            }
        }
        let suffix = std::str::from_utf8_unchecked(&bytes.get_unchecked(skip..));

        (result, suffix)
    }
}

pub fn save_vocab(
    vocab: &[(impl AsRef<str>, (u64, f64))],
    path: impl AsRef<Path>,
) -> io::Result<()> {
    info!(
        "Writing vocabulary with {} distinct words to {} ...",
        vocab.len(),
        path.as_ref().display()
    );

    let mut output = BufWriter::new(fs::File::create(path)?);
    writeln!(output, "WORD\tTOTAL_COUNT\tWEIGHTED_FREQUENCY")?;

    for item in vocab.iter() {
        writeln!(
            output,
            "{}\t{}\t{}",
            item.0.as_ref(),
            (item.1).0,
            (item.1).1
        )?;
    }
    Ok(())
}

pub fn load_vocab(
    path: impl AsRef<Path>,
    max_size: Option<u32>,
) -> Result<HashMap<String, u32, FnvBuildHasher>, Box<dyn Error>> {
    let mut input = io::BufReader::new(fs::File::open(path)?);
    let mut header = String::new();
    input.read_line(&mut header)?;
    assert_eq!(header, "WORD\tTOTAL_COUNT\tWEIGHTED_FREQUENCY\n");

    let mut vocab = if let Some(max_size) = max_size {
        HashMap::with_capacity_and_hasher(max_size as usize, FnvBuildHasher::default())
    } else {
        HashMap::with_hasher(FnvBuildHasher::default())
    };
    let mut buf = Vec::<u8>::new();
    while Some(vocab.len() as u32) != max_size && input.read_until(b'\t', &mut buf)? != 0 {
        let word = std::str::from_utf8(&buf[..buf.len() - 1])?.to_string();
        assert!(vocab.insert(word, vocab.len() as u32).is_none());
        input.read_until(b'\n', &mut buf)?; // Why is there no `BufRead::skip_until()`?
        buf.clear();
    }

    if let Some(max_size) = max_size {
        if vocab.len() < max_size as usize {
            warn!(
                "Requested vocabulary size {} but found only {} entries in vocabulary file.",
                max_size,
                vocab.len()
            );
        }
    }

    info!("Vocabulary size: {}", vocab.len());

    Ok(vocab)
}
