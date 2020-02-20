# Preprocessing of the Google Books Corpus

This directory contains the code of three small Rust programs that parse 5-gram
files from the [Google Books corpus] and generate the `n^+_t` matrices that our
models expect.

The preprocessing pipeline goes as follows:

1. Install Rust and Cargo if you don't already have them on your system:
    [https://rustup.rs/](https://rustup.rs/)

2. Compile all tools in this directory:

    ```bash
    cargo build --release
    ```

    This will take a minute or two, and it will place the resulting binaries in a
    newly created directory `target/release`.

3. Download all 1-gram files from the [Google Books corpus] and place them in a
    directory `1grams`. Do *not* extract the gzip compressed files. Then run

    ```bash
     target/release/mk_vocab --from 1980 --to 2008 --vocab_size 100000 1grams
    ```

    Here, you may want to replace `1980` and `2008` with other years if you are
    interested in a different range of years, and you may want to choose a
    different vocabulary size. This will read in the 1-grams, sort them by
    average frequency, and write out three tab separated files to the directory
    `1grams`:
    - `vocab_1800to2008_vocabsize100000.tsv`:
      The vocabulary, i.e., a list of the 100,000 most frequent words. More
      precisely, the list is sorted by average word frequency (averaged over the
      specified range of years). This is the main output file of this
      preprocessing step. The other files are just for your information.
    - `counts_1800to2008_vocabsize100000.tsv`:
      Table of total number of words per year in the corpus.
    - `vocab_1800to2008_vocabsize100000_unweighted.tsv`
      Alternative vocabulary, sorted by total counts rather than average
      frequency. This is the vocabulary we will be using here.

4. Download all 5-gram files from the [Google Books corpus] and place them in a
    directory `5grams` (takes >200 GB of disk space). *Do not extract the gzip
    compressed files.* Then run

    ```bash
    mkdir by_year
    target/release/sort_by_year --from 1980 --to 2008 \
         --vocab 1grams/vocab_1800to2008_vocabsize100000_unweighted.tsv 5grams by_year
    ```

    This is the most computationally expensive step of the preprocessing
    pipeline. It will probably run for a day or two, but it won't need a lot of
    memory. This step essentially transposes the gigantic data set: the 5-grams
    in the Google books corpus are sorted alphabetically, but we need them
    sorted by year. The `sort_by_year` program generates a compressed file for
    each year in the range of interest and writes it to the directory `by_year`.
    In doing so, it also replaces words by their indices in the vocabulary.
    By default, the output files in this step are only weakly compressed because
    the resulting files are temporary and can be removed after step 5 below.
    If you're tight on disk space, you can run the command with argument
    `--compression_level 9` (will take much longer).

5. Generate the final sufficient statistics `n^+_t` by running the command:

    ```bash
    mkdir npos
    for year in `seq 1980 2008`; do
         target/release/mk_npos --year $year by_year npos
    done
    ```

    This iterates over each year, reads in the file generated in step 4 above,
    calculates the sufficient statistics from it, and writes it to a compressed
    file in the directory `npos`. Do not extract the gzip compressed output
    files, the model extracts them in-memory on startup. The files contain a CSR
    encoded sparse matrix of word co-occurrences for each year (weighted by
    closeness in the corpus). See the model files for an example how to read in
    files in this format. The rows and columns in the matrices correspond to the
    words in the vocabulary, in the same order.

    **NOTE:** The first line in the vocabulary file is a header, so the n'th
    row and column of a co-occurrence matrix correspond to the (n+1)'th line in
    the vocabulary file.

    This step of the preprocessing pipeline treats each year independently from
    the others, so the loop in the above shell command can be parallelized (it
    is most likely CPU bound because we use relatively strong compression by
    default here).

6.  All files except the vocabulary and the files in the directory `npos` can be
    removed at this point.

7.  To generate a single dat set that combines all years, run:
    ```bash
    python combine_datasets.py \
        'npos/npos_year%d.bin.gz' \
        npos/npos_years1980-2008.bin.gz \
        --first 1980 --last 2008
    ```

[Google Books corpus]: http://storage.googleapis.com/books/ngrams/books/datasetsv2.html
