use std::convert::TryFrom;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use fxhash::FxHashMap;
use log::info;
use sprs;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(
    about = r#"Generate a sufficient statistics (n^+ matrix) for each year.

Run this tool after generating binary 5-gram files with the tool `sort_by_year`.
This tool reads in a file "5gram_year<YEAR>.bin.gz" in the specified input
directory, calculates a sufficient statistics for training Dynamic Word
Embeddings, and writes it to a file "npos_year<YEAR>.bin.gz" in the specified
output directory. Several instances of this tool may be run in parallel for
different years if enough memory is available. Note that too many parallel jobs
on the same computer may actually slow down overall progress because a large
part of the calculation is I/O bound."#
)]

struct Opt {
    /// The year for which to calculate the sufficient statistics.
    #[structopt(long)]
    year: i32,

    /// Input directory.
    pub input_dir: PathBuf,

    /// Output directory.
    output_dir: PathBuf,

    /// Compression level of generated output.
    ///
    /// Must be a number from 1 to 9, with 1 being fastest and 9 being strongest
    /// compression. Empirically, changing the compression level from 1 to 9 saves
    /// about 15% of space but compression slows down drastically for levels higher
    /// than about 5. The compression level has little effect on decompression speed
    /// albeit decompression is a bit slow for levels lower than 3 or higher than 7.
    ///
    /// The default compression level here is rather high because this tool is the
    /// final step in the preprocessing pipeline, so the resulting files are likely
    /// going to be kept around for an extended time and copied across networks.
    /// Also, this tool writes out only a small amount of data compared to how much
    /// it reads in, so execution time is mainly bound on read speed anyway.
    #[structopt(long, short, name = "1-9", default_value = "7")]
    compression_level: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let Opt {
        input_dir,
        output_dir,
        ..
    } = opt;
    let mut input_path = input_dir;
    input_path.push(format!("5grams_year{}.bin.gz", opt.year));
    info!("Reading from file {} ...", input_path.display());
    let file = fs::File::open(input_path)?;
    let mut input = BufReader::new(GzDecoder::new(file));
    let vocab_size = input.read_u32::<LittleEndian>()?;

    let mut matrix = FxHashMap::<(u32, u32), u64>::default();

    let mut words = [0u32; 5];
    loop {
        input.read_u32_into::<LittleEndian>(&mut words)?;
        let total_count = input.read_u32::<LittleEndian>()?;
        let book_count = input.read_u32::<LittleEndian>()?;

        if total_count == 0 {
            // Assert that this is the special EOF marker.
            assert_eq!(book_count, 0);
            assert_eq!(words, [std::u32::MAX - 1; 5]);
            // Assert that it's indeed the end of the underlying file.
            assert_eq!(input.read(&mut [0u8; 1])?, 0);
            break;
        }

        for (i, &word_i) in words.iter().enumerate().skip(1) {
            if word_i < vocab_size {
                for (j, &word_j) in words[..i].iter().enumerate() {
                    if word_j < vocab_size {
                        let weight = 5 - i as u32 + j as u32;
                        let weighted_count = u64::from(weight * total_count);
                        let (first, second) = if word_i < word_j {
                            (word_i, word_j)
                        } else {
                            (word_j, word_i)
                        };
                        matrix
                            .entry((first, second))
                            .and_modify(|v| *v += weighted_count)
                            .or_insert(weighted_count);
                    }
                }
            }
        }
    }

    info!(
        "Converting matrix for year {} into CSR format ...",
        opt.year
    );
    let orig_num_entries = matrix.len();
    let mut triplet_matrix = sprs::TriMatI::<f32, u32>::with_capacity(
        (vocab_size as usize, vocab_size as usize),
        2 * orig_num_entries,
    );

    for ((i, j), val) in matrix {
        let val = val as f32;
        if i == j {
            triplet_matrix.add_triplet(i as usize, j as usize, val / 2f32);
        } else {
            triplet_matrix.add_triplet(i as usize, j as usize, val / 4f32);
            triplet_matrix.add_triplet(j as usize, i as usize, val / 4f32);
        }
    }
    let csr_matrix = triplet_matrix.to_csr();

    let mut path = output_dir;
    path.push(format!("npos_year{}.bin.gz", opt.year));
    info!(
        "Saving {} nonzero entries to file {} ...",
        csr_matrix.nnz(),
        path.display()
    );
    let file = fs::File::create(path)?;
    let output = GzEncoder::new(file, Compression::new(opt.compression_level));
    let mut output = BufWriter::new(output);

    let (indptr, indices, data) = csr_matrix.into_raw_storage();
    assert_eq!(indices.len(), data.len());
    assert_eq!(indptr[indptr.len() - 1] as usize, data.len());
    assert!(indptr.len() <= vocab_size as usize + 1);
    assert!(indices.len() >= orig_num_entries);
    assert!(indices.len() <= 2 * orig_num_entries + 1); // `+1` for empty matrices.

    output.write_u32::<LittleEndian>(vocab_size)?;
    output.write_u32::<LittleEndian>(u32::try_from(data.len())?)?;
    let indptr_len = u32::try_from(indptr.len())?;
    let dat_len = u32::try_from(data.len())?;

    for i in indptr {
        output.write_u32::<LittleEndian>(i)?;
    }
    for _ in indptr_len..=vocab_size {
        // Usually, the `indptr` in a CSR matrix has length `rows + 1`. However,
        // `sprs::CsMatBase` uses a shorter `indptr` if the last rows of the matrix
        // have no nonzero entries. This confuses python's `scipy.sparse`, so we
        // pad `indptr` to the length that `scipy.sparse` expects.
        output.write_u32::<LittleEndian>(dat_len)?;
    }

    for i in indices {
        output.write_u32::<LittleEndian>(i)?;
    }
    for v in data {
        output.write_f32::<LittleEndian>(v)?;
    }

    info!("Done (year {}).", opt.year);
    Ok(())
}
