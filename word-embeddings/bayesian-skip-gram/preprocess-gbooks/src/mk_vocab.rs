use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use fnv::FnvBuildHasher;
use log::{info, warn};
use structopt::StructOpt;

use google_books::{self, iterate_ngram_files, save_vocab};

#[derive(StructOpt)]
#[structopt(
    about = r#"Parse Google Books 1-gram files and create a vocabulary sorted by frequency.

This tool reads all `*.gz` files in the input directory and writes three tab
separated files to the output directory. The first line in each output file is a
header, and we strictly enforce UTF-8 encoding throughout all three output
files. In detail, the following three files are generated:

  * vocab_<FIRSTYEAR>to<LASTYEAR>_vocabsize<VOCABSIZE>.tsv:
    The main output file. A list of words that occur in the scanned corpus
    within the range of years specified by the parameters --from and --to.
    Each line has the format "<word><TAB><count><TAB><avg_frequency><LF>".
    Here, <count> is the total number of occurrences of <word> in the requested
    range of years, <avg_frequency> is the frequency of the word, averaged over
    all years in the requested range, <TAB> is 0x09 in UTF-8 encoding and <LF>
    is line feed (0x0a). The list is sorted by <avg_frequency>. Note that the
    frequencies typically don't add up to 1 because we cut off the vocabulary
    after the <VOCABSIZE> most frequent words, which can be controlled by the
    command line parameter --vocab-size.

  * vocab_<FIRSTYEAR>to<LASTYEAR>_vocabsize<VOCABSIZE>_unweighted.tsv:
    Similar to the above file, but sorted by <count>. Note that sorting is done
    before the truncation to <VOCABSIZE> words, so the two "vocab_*" files
    typically contain different words. You usually do NOT want to use a
    vocabulary sorted by counts because the corpus contains so much more recent
    data compared to historic data that sorting by absolute word counts leads to
    a vocabulary that lacks many important historic words. This file is only
    generated for completeness since a lot of heavy work that's necessary to
    create this file has to be done anyway.

  * counts_<FIRSTYEAR>to<LASTYEAR>_vocabsize<VOCABSIZE>.tsv:
    A table of the total number of words per year in the corpus. Note that the
    counts in this file are over all words, even the ones that are omitted from
    the "vocab_*" files due to truncation to <VOCABSIZE> words."#
)]

struct Opt {
    #[structopt(flatten)]
    common: google_books::Opt,

    /// Vocabulary size.
    #[structopt(long, default_value = "100000")]
    vocab_size: u32,

    /// Output directory. [defaults to input directory].
    output_dir: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let word_counts = read_1grams(&opt)?;
    let word_stats = summary_stats(&opt, word_counts)?;
    sort_and_save_vocab(&opt, word_stats)?;

    info!("Done.");
    Ok(())
}

fn read_1grams(opt: &Opt) -> io::Result<Vec<HashMap<String, u32, FnvBuildHasher>>> {
    assert!(opt.common.from <= opt.common.to);
    let num_years = (opt.common.to - opt.common.from + 1) as usize;
    let mut word_counts = vec![HashMap::with_hasher(FnvBuildHasher::default()); num_years];

    for n_gram_file in iterate_ngram_files(&opt.common)? {
        let mut n_gram_file = n_gram_file?;
        loop {
            match n_gram_file.next([""; 1])? {
                google_books::Event::NGram(words, stats) => {
                    let dict = &mut word_counts[(stats.year - opt.common.from) as usize];

                    // Don't use entry API because it allocates even if entry already exists.
                    if let Some(c) = dict.get_mut(words[0]) {
                        *c += stats.total_count;
                    } else {
                        dict.insert(words[0].to_string(), stats.total_count);
                    }
                }
                google_books::Event::Skip => {}
                google_books::Event::EndOfFile => break,
            }
        }
    }

    info!("Done reading files.");
    Ok(word_counts)
}

fn summary_stats(
    opt: &Opt,
    word_counts: Vec<HashMap<String, u32, FnvBuildHasher>>,
) -> io::Result<HashMap<String, (u64, f64)>> {
    let file_name = format!(
        "counts_{}to{}_vocabsize{}.tsv",
        opt.common.from, opt.common.to, opt.vocab_size,
    );
    let output_path = get_output_path(&opt, &file_name);
    info!(
        "Writing numbers of words per year to {} ...",
        output_path.display()
    );

    let mut counts_output = BufWriter::new(fs::File::create(output_path)?);
    writeln!(counts_output, "YEAR\tNUMBER_OF_WORDS")?;

    assert!(opt.common.from <= opt.common.to);
    let num_years = (opt.common.to - opt.common.from + 1) as u64;
    let normalizers = word_counts
        .iter()
        .zip(opt.common.from..=opt.common.to)
        .map(|(dict, year)| {
            let counts = dict.values().map(|c| u64::from(*c)).sum::<u64>();
            writeln!(counts_output, "{}\t{}", year, counts).unwrap();
            1f64 / (counts * num_years) as f64
        })
        .collect::<Vec<_>>();

    drop(counts_output);

    info!("Calculating statistics ...");
    let max_vocab_size_per_year = word_counts.iter().map(|dict| dict.len()).max().unwrap();
    let capacity_guess = 2 * max_vocab_size_per_year as usize;
    let mut word_stats = HashMap::<String, (u64, f64)>::with_capacity(capacity_guess);
    for (year_dict, normalizer) in word_counts.into_iter().zip(normalizers) {
        for (word, &count) in year_dict.iter() {
            let weighted = normalizer * f64::from(count);
            // Don't use entry API because it allocates even if entry already exists.
            if let Some((c, w)) = word_stats.get_mut(word) {
                *c += u64::from(count);
                *w += weighted;
            } else {
                word_stats.insert(word.to_string(), (u64::from(count), weighted));
            }
        }
    }
    info!("Found {} distinct words in total.", word_stats.len());
    Ok(word_stats)
}

fn sort_and_save_vocab(opt: &Opt, word_stats: HashMap<String, (u64, f64)>) -> io::Result<()> {
    let mut word_stats = word_stats.into_iter().collect::<Vec<_>>();
    let vocab_size = if opt.vocab_size as usize <= word_stats.len() {
        opt.vocab_size
    } else {
        warn!(
            "Requested vocabulary size of {} but found only {} distinct words.",
            opt.vocab_size,
            word_stats.len()
        );
        word_stats.len() as u32
    };

    info!("Sorting vocabulary by (unweighted) total number of occurrences ...");
    word_stats.sort_by(|(w1, (c1, _)), (w2, (c2, _))| (c2, w2).cmp(&(c1, w1)));
    info!("Highest count: {}", (word_stats.first().unwrap().1).0);
    info!(
        "Lowest included count: {}",
        (word_stats[std::cmp::min(word_stats.len(), opt.vocab_size as usize) - 1].1).0
    );
    info!("Lowest count: {}", (word_stats.last().unwrap().1).0,);

    let file_name = format!(
        "vocab_{}to{}_vocabsize{}_unweighted.tsv",
        opt.common.from, opt.common.to, opt.vocab_size,
    );
    save_vocab(
        &word_stats[..vocab_size as usize],
        get_output_path(&opt, &file_name),
    )?;

    info!("Sorting vocabulary by word frequencies (averaged over all years) ...");
    word_stats.sort_by(|(w1, (_, f1)), (w2, (_, f2))| (f2, w2).partial_cmp(&(f1, w1)).unwrap());
    info!("Highest frequency: {}", (word_stats.first().unwrap().1).1);
    info!(
        "Lowest included frequency: {}",
        (word_stats[std::cmp::min(word_stats.len(), opt.vocab_size as usize) - 1].1).1
    );
    info!("Lowest frequency: {}", (word_stats.last().unwrap().1).1,);

    let file_name = format!(
        "vocab_{}to{}_vocabsize{}.tsv",
        opt.common.from, opt.common.to, opt.vocab_size,
    );
    save_vocab(
        &word_stats[..vocab_size as usize],
        get_output_path(&opt, &file_name),
    )?;
    Ok(())
}

fn get_output_path(opt: &Opt, file_name: &str) -> PathBuf {
    let mut path = opt
        .output_dir
        .as_ref()
        .unwrap_or_else(|| &opt.common.input_dir)
        .clone();
    path.push(file_name);
    path
}
