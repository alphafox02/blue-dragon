// Copyright 2025-2026 CEMAXECUTER LLC

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use crossbeam::channel::Sender;
use crate::{SampleBuf, SdrSource};

/// IQ sample format for file input
#[derive(Debug, Clone, Copy)]
pub enum SampleFormat {
    /// Complex int8 (CS8): pairs of i8
    Ci8,
    /// Complex int16 (CS16): pairs of i16, little-endian
    Ci16,
    /// Complex float32 (CF32): pairs of f32, little-endian
    Cf32,
}

/// IQ file reader: reads samples from a file and sends them as SampleBuf blocks.
pub struct FileSource {
    path: String,
    format: SampleFormat,
    sample_rate: u32,
    center_freq: u64,
    /// Number of complex samples per block
    block_size: usize,
    running: bool,
}

impl FileSource {
    pub fn new(
        path: impl Into<String>,
        format: SampleFormat,
        sample_rate: u32,
        center_freq: u64,
    ) -> Self {
        Self {
            path: path.into(),
            format,
            sample_rate,
            center_freq,
            block_size: 65536, // 64K complex samples per block
            running: false,
        }
    }

    pub fn set_block_size(&mut self, size: usize) {
        self.block_size = size;
    }

    /// Read a block of samples from the file, converting to int16 IQ pairs.
    fn read_block_ci8(reader: &mut BufReader<File>, num_samples: usize) -> io::Result<Option<Vec<i16>>> {
        let bytes_needed = num_samples * 2; // 2 bytes per complex sample (I, Q)
        let mut buf = vec![0u8; bytes_needed];
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(None);
        }
        let actual_samples = n / 2;
        let mut out = Vec::with_capacity(actual_samples * 2);
        for i in 0..actual_samples {
            // Scale i8 [-128, 127] to i16 range by shifting left 8
            out.push((buf[i * 2] as i8 as i16) << 8);
            out.push((buf[i * 2 + 1] as i8 as i16) << 8);
        }
        Ok(Some(out))
    }

    fn read_block_ci16(reader: &mut BufReader<File>, num_samples: usize) -> io::Result<Option<Vec<i16>>> {
        let bytes_needed = num_samples * 4; // 4 bytes per complex sample
        let mut buf = vec![0u8; bytes_needed];
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(None);
        }
        let actual_samples = n / 4;
        let mut out = Vec::with_capacity(actual_samples * 2);
        for i in 0..actual_samples {
            let base = i * 4;
            let i_val = i16::from_le_bytes([buf[base], buf[base + 1]]);
            let q_val = i16::from_le_bytes([buf[base + 2], buf[base + 3]]);
            out.push(i_val);
            out.push(q_val);
        }
        Ok(Some(out))
    }

    fn read_block_cf32(reader: &mut BufReader<File>, num_samples: usize) -> io::Result<Option<Vec<i16>>> {
        let bytes_needed = num_samples * 8; // 8 bytes per complex sample
        let mut buf = vec![0u8; bytes_needed];
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(None);
        }
        let actual_samples = n / 8;
        let mut out = Vec::with_capacity(actual_samples * 2);
        for i in 0..actual_samples {
            let base = i * 8;
            let i_f = f32::from_le_bytes([buf[base], buf[base + 1], buf[base + 2], buf[base + 3]]);
            let q_f = f32::from_le_bytes([buf[base + 4], buf[base + 5], buf[base + 6], buf[base + 7]]);
            // Convert float [-1, 1] to int16
            out.push((i_f * 32767.0).clamp(-32768.0, 32767.0) as i16);
            out.push((q_f * 32767.0).clamp(-32768.0, 32767.0) as i16);
        }
        Ok(Some(out))
    }
}

impl SdrSource for FileSource {
    fn start(&mut self, tx: Sender<SampleBuf>) -> Result<(), String> {
        let path = Path::new(&self.path);
        let file = File::open(path).map_err(|e| format!("failed to open {}: {}", self.path, e))?;
        let mut reader = BufReader::with_capacity(1024 * 1024, file);

        self.running = true;
        log::info!(
            "reading IQ from {} ({:?}, {} Hz, {} MHz)",
            self.path,
            self.format,
            self.sample_rate,
            self.center_freq / 1_000_000
        );

        while self.running {
            let result = match self.format {
                SampleFormat::Ci8 => Self::read_block_ci8(&mut reader, self.block_size),
                SampleFormat::Ci16 => Self::read_block_ci16(&mut reader, self.block_size),
                SampleFormat::Cf32 => Self::read_block_cf32(&mut reader, self.block_size),
            };

            match result {
                Ok(Some(data)) => {
                    let num_samples = data.len() / 2;
                    if tx.send(SampleBuf { data, num_samples }).is_err() {
                        break; // receiver dropped
                    }
                }
                Ok(None) => {
                    log::info!("end of file: {}", self.path);
                    break;
                }
                Err(e) => {
                    return Err(format!("read error: {}", e));
                }
            }
        }

        Ok(())
    }

    fn stop(&mut self) {
        self.running = false;
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn center_frequency(&self) -> u64 {
        self.center_freq
    }
}
