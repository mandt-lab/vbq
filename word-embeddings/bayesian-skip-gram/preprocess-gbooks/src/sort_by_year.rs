use std::error::Error;
use std::fs;
use std::io::BufWriter;
use std::path::PathBuf;

use byteorder::{LittleEndian, WriteBytesExt};
use flate2::{write::GzEncoder, Compression};
use log::info;
use structopt::StructOpt;

use google_books::{self, iterate_ngram_files, load_vocab};

#[derive(StructOpt)]
#[structopt(
    about = r#"Sort data in 5-gram files by year (using on-disk radix sort).

Reads in all 5-gram files of the corpus in compressed form and generates a
compressed binary output file for each year within the region specified by the
parameters --from and --to. The conversion from text to binary is specified in
a vocabulary file that can be obtained with the `mk_vocab` tool.

The transformation done by this tool is necessary because the Google Books
corpus stores N-grams sorted primarily by alphabet, and only secondarily by
year. By contrast, Dynamic Word Embeddings are trained on a sufficient
statistics *per year*. This tool sorts the N-grams by year. A sufficient
statistics per year can then be generated with the tool `mk_npos`. The reason
why we split up the sorting and the calculation of the sufficient statistics is
because doing both at once would require a lot of memory, as a one would have to
keep around an editable form of all sufficient statistics at the same time. By
contrast, after the data is sorted by year, we can calculate the sufficient
statistics for each year independently, so memory requirements are independent
of the number of processed years. The sorting itself requires hardly any memory
because we do radix sort and we write out data to the appropriate compressed
file almost immediately after reading it (up to buffering)."#
)]

struct Opt {
    #[structopt(flatten)]
    common: google_books::Opt,

    /// Path to stored a vocabulary (create vocabulary with `mk_vocab`)
    #[structopt(long)]
    vocab: PathBuf,

    /// Output directory.
    output_dir: PathBuf,

    /// Compression level of generated output.
    ///
    /// Must be a number from 1 to 9, with 1 being fastest and 9 being strongest
    /// compression. Empirically, changing the compression level from 1 to 9 saves
    /// about 15% of space but compression slows down drastically for levels higher
    /// than about 5. The compression level has little effect on decompression speed
    /// albeit it's a bit slow for levels lower than 3 or higher than 7.
    ///
    /// The default compression level here is rather low because the files generated
    /// at this stage in the preprocessing pipeline are considered somewhat
    /// ephemeral as they can be discarded after running `mk_npos`.
    #[structopt(long, short, name = "1-9", default_value = "4")]
    compression_level: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    assert!(opt.common.from <= opt.common.to);
    let vocab = load_vocab(opt.vocab, None)?;
    let output_dir = &opt.output_dir.clone();
    let compression_level = opt.compression_level;

    let mut outputs = (opt.common.from..=opt.common.to)
        .map(|year| {
            let mut path = output_dir.clone();
            path.push(format!("5grams_year{}.bin.gz", year));
            let file = fs::File::create(path).expect("Cannot create output file.");
            let output = GzEncoder::new(file, Compression::new(compression_level));
            let mut output = BufWriter::new(output);
            output
                .write_u32::<LittleEndian>(vocab.len() as u32)
                .expect("Cannot write to output file.");
            output
        })
        .collect::<Vec<_>>();

    for n_gram_file in iterate_ngram_files(&opt.common)? {
        let mut n_gram_file = n_gram_file?;
        loop {
            match n_gram_file.next([""; 5])? {
                google_books::Event::NGram(words, stats) => {
                    let out = &mut outputs[(stats.year - opt.common.from) as usize];
                    for word in words.iter() {
                        out.write_u32::<LittleEndian>(
                            vocab.get(*word).cloned().unwrap_or(std::u32::MAX),
                        )?;
                    }
                    out.write_u32::<LittleEndian>(stats.total_count)?;
                    out.write_u32::<LittleEndian>(stats.book_count)?;
                }
                google_books::Event::Skip => {}
                google_books::Event::EndOfFile => break,
            }
        }
    }

    for mut output in outputs {
        // Write a special EOF marker (recognizable by zero `total_count` and `book_count`).
        for _ in 0..5 {
            output.write_u32::<LittleEndian>(std::u32::MAX - 1)?;
        }
        output.write_u32::<LittleEndian>(0)?;
        output.write_u32::<LittleEndian>(0)?;
    }

    info!("Done.");
    Ok(())
}
