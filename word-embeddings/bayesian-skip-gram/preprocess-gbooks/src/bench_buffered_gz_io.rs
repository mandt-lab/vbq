//! This binary benchmarks the performance of reading and writing `*.gz` files
//! with different variations of buffering:
//! - unbuffered
//! - wrapping the uncompressed stream in a `BufReader` / `BufWriter`
//! - wrapping the compressed stream in a `BufReader` / `BufWriter`
//! - wrapping both in a `BufReader` / `BufWriter`
//!
//! It seems like we get best performance
//! - for reading: using `BufReader<flate2::read::GzDecoder<std::fs::File>>`
//! - for writing: using `BufWriter<flate2::write::GzEncoder<std::fs::File>>`
//! i.e., one should wrap the `GzDecoder` / `GzEncoder` in a
//! `BufReader` / `BufWriter` but not the wrapped file.
//!
//! It would be nicer use the `criterion` crate for these benchmarks, but I
//! couldn't get that to work without a lot of boilerplate code.

use std::fs::{remove_file, File};
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use flate2::{write::GzEncoder, Compression};
use fxhash::FxHasher;
use time::precise_time_ns;

pub fn bench_function(label: &str, f: impl Fn()) {
    let times = (0..10)
        .map(|_| {
            let start = precise_time_ns();
            f();
            (precise_time_ns() - start) as f64 * 1e-9
        })
        .collect::<Vec<_>>();

    let sum = times.iter().sum::<f64>();
    let mean = sum / times.len() as f64;
    let sum_square = times.iter().map(|t| t * t).sum::<f64>();
    let variance = (sum_square - sum * sum / times.len() as f64) / (times.len() - 1) as f64;
    println!("  - {}: {} +- {} s", label, mean, variance.sqrt());
}

pub fn base_write<W, B>(uncompressed: &[[u32; 4]], builder: B, label: &str)
where
    W: Write,
    B: Fn(File) -> W,
{
    bench_function(label, || {
        let file = File::create("tmp_file_can_be_removed.gz").unwrap();
        let mut writer = builder(file);
        for block in uncompressed {
            writer.write_u32::<LittleEndian>(block[0]).unwrap();
            writer.write_u32::<LittleEndian>(block[1]).unwrap();
            writer.write_u32::<LittleEndian>(block[2]).unwrap();
            writer.write_u32::<LittleEndian>(block[3]).unwrap();
        }
    });
    remove_file("tmp_file_can_be_removed.gz").unwrap_or(());
}

pub fn base_read<R, B>(uncompressed: &[[u32; 4]], builder: B, label: &str)
where
    R: Read,
    B: Fn(File) -> R,
{
    let hash = {
        let file = File::create("tmp_file_can_be_removed.gz").unwrap();
        let mut writer = BufWriter::new(GzEncoder::new(file, Compression::default()));
        let mut hasher = FxHasher::default();
        for block in uncompressed {
            block[0].hash(&mut hasher);
            block[1].hash(&mut hasher);
            block[2].hash(&mut hasher);
            block[3].hash(&mut hasher);
            writer.write_u32::<LittleEndian>(block[0]).unwrap();
            writer.write_u32::<LittleEndian>(block[1]).unwrap();
            writer.write_u32::<LittleEndian>(block[2]).unwrap();
            writer.write_u32::<LittleEndian>(block[3]).unwrap();
        }
        hasher.finish()
    };

    bench_function(label, || {
        let file = File::open("tmp_file_can_be_removed.gz").unwrap();
        let mut reader = builder(file);
        let mut hasher = FxHasher::default();
        while let Ok(val) = reader.read_u32::<LittleEndian>() {
            val.hash(&mut hasher);
        }
        assert_eq!(hasher.finish(), hash);
    });
    remove_file("tmp_file_can_be_removed.gz").unwrap_or(());
}

macro_rules! bench_method {
    ($name:ident, $method:ident, $base:ident, $builder:expr) => {
        println!("- Method `{}` ...", stringify!($method));
        {
            let uncompressed = generate_low_entropy_data();
            $base(&uncompressed, $builder, "low_entropy");
        }

        {
            let uncompressed = generate_medium_entropy_data();
            $base(&uncompressed, $builder, "medium_entropy");
        }

        {
            let uncompressed = generate_high_entropy_data();
            $base(&uncompressed, $builder, "high_entropy");
        }
    };
}

macro_rules! bench_task {
    ($name:ident, $base:ident, $raw_builder:expr, $buf_builder:expr) => {
        println!("Benchmarking task `{}` ...", stringify!($name));
        bench_method! { $name, raw, $base, $raw_builder }
        bench_method! {
            $name,
            outer_buf,
            $base,
            |file| $buf_builder($raw_builder(file))
        }
        bench_method! {
            $name,
            inner_buf,
            $base,
            |file| $raw_builder($buf_builder(file))
        }
        bench_method! {
            $name,
            both_buf,
            $base,
            |file| $buf_builder($raw_builder($buf_builder(file)))
        }
    };
}

fn main() {
    bench_task! {
        encode,
        base_write,
        |file| flate2::write::GzEncoder::new(file, flate2::Compression::default()),
        std::io::BufWriter::new
    }

    bench_task! {
        decode,
        base_read,
        flate2::read::GzDecoder::new,
        std::io::BufReader::new
    }
}

/// Can be compressed to about 19%.
fn generate_low_entropy_data() -> Vec<[u32; 4]> {
    const NUM_BLOCKS: u32 = 1 << 20;
    (1..=NUM_BLOCKS)
        .map(|i| {
            [
                i.wrapping_mul(i & 1),
                i.wrapping_mul((i >> 1) & 1),
                i.wrapping_mul((i >> 2) & 1),
                i.wrapping_mul((i >> 3) & 1),
            ]
        })
        .collect()
}

/// Can be compressed to about 35%.
fn generate_medium_entropy_data() -> Vec<[u32; 4]> {
    const NUM_BLOCKS: u32 = 1 << 20;
    (1..=NUM_BLOCKS)
        .map(|i| {
            [
                i.wrapping_mul(i & 3),
                i.wrapping_mul((i >> 1) & 3),
                i.wrapping_mul((i >> 2) & 3),
                i.wrapping_mul((i >> 3) & 3),
            ]
        })
        .collect()
}

/// Can be compressed to about 86%.
fn generate_high_entropy_data() -> Vec<[u32; 4]> {
    const NUM_BLOCKS: u32 = 1 << 20;
    let mut hasher = FxHasher::default();
    (1..=NUM_BLOCKS)
        .map(|i| {
            i.hash(&mut hasher);
            let noise = hasher.finish() as u32;
            [
                noise.wrapping_mul(0x4EB7_FE4E + (i & 0x0101_0101)),
                noise.wrapping_mul(0x4EB7_FE4E + ((i >> 1) & 0x0101_0101)),
                noise.wrapping_mul(0x4EB7_FE4E + ((i >> 2) & 0x0101_0101)),
                noise.wrapping_mul(0x4EB7_FE4E + ((i >> 3) & 0x0101_0101)),
            ]
        })
        .collect()
}
