// Copyright 2025-2026 CEMAXECUTER LLC

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use bd_dsp::burst::Burst;
use bd_protocol::Timespec;
use num_complex::Complex32;

const FILE_MAGIC: &[u8; 8] = b"BDBURST\0";
const RECORD_MAGIC: &[u8; 4] = b"BRST";
const VERSION: u16 = 1;
const FILE_HEADER_LEN: u16 = 16;
const RECORD_HEADER_LEN: u32 = 48;
const CHANNEL_SAMPLE_RATE: u32 = 2_000_000;
const FLAG_SCAN: u32 = 1;
const MAX_SAMPLES_PER_RECORD: usize = 1_048_576;

pub type FileBurstWriter = BurstWriter<BufWriter<File>>;
pub type FileBurstReader = BurstReader<BufReader<File>>;

pub struct BurstWriter<W: Write> {
    inner: W,
    bytes_written: u64,
    max_bytes: u64,
}

impl FileBurstWriter {
    pub fn create(path: &Path, max_bytes: u64) -> io::Result<Self> {
        let file = File::create(path)?;
        BurstWriter::new(BufWriter::new(file), max_bytes)
    }
}

impl<W: Write> BurstWriter<W> {
    pub fn new(mut inner: W, max_bytes: u64) -> io::Result<Self> {
        inner.write_all(FILE_MAGIC)?;
        inner.write_all(&VERSION.to_le_bytes())?;
        inner.write_all(&FILE_HEADER_LEN.to_le_bytes())?;
        inner.write_all(&CHANNEL_SAMPLE_RATE.to_le_bytes())?;
        Ok(Self {
            inner,
            bytes_written: FILE_HEADER_LEN as u64,
            max_bytes,
        })
    }

    /// Returns false when this record would exceed the configured file limit.
    pub fn write_burst(&mut self, burst: &Burst) -> io::Result<bool> {
        if burst.samples.len() > MAX_SAMPLES_PER_RECORD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "burst is too large",
            ));
        }
        let sample_bytes = burst
            .samples
            .len()
            .checked_mul(4)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "burst is too large"))?;
        let record_len = RECORD_HEADER_LEN as u64 + sample_bytes as u64;
        if self.max_bytes != 0 && self.bytes_written + record_len > self.max_bytes {
            return Ok(false);
        }

        let peak = burst.samples.iter().fold(0.0f32, |acc, sample| {
            acc.max(sample.re.abs()).max(sample.im.abs())
        });
        let scale = if peak > 0.0 {
            peak / i16::MAX as f32
        } else {
            1.0
        };
        let flags = if burst.scan { FLAG_SCAN } else { 0 };

        self.inner.write_all(RECORD_MAGIC)?;
        self.inner.write_all(&(record_len as u32).to_le_bytes())?;
        self.inner
            .write_all(&burst.timestamp.tv_sec.to_le_bytes())?;
        self.inner
            .write_all(&(burst.timestamp.tv_nsec as u32).to_le_bytes())?;
        self.inner.write_all(&burst.freq.to_le_bytes())?;
        self.inner.write_all(&burst.rssi_db.to_le_bytes())?;
        self.inner.write_all(&burst.noise_db.to_le_bytes())?;
        self.inner.write_all(&burst.num.to_le_bytes())?;
        self.inner
            .write_all(&(burst.samples.len() as u32).to_le_bytes())?;
        self.inner.write_all(&flags.to_le_bytes())?;
        self.inner.write_all(&scale.to_le_bytes())?;
        for sample in &burst.samples {
            let i = (sample.re / scale)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let q = (sample.im / scale)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            self.inner.write_all(&i.to_le_bytes())?;
            self.inner.write_all(&q.to_le_bytes())?;
        }
        self.bytes_written += record_len;
        Ok(true)
    }

    #[cfg(test)]
    fn into_inner(self) -> W {
        self.inner
    }
}

pub struct BurstReader<R: Read> {
    inner: R,
}

impl FileBurstReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        BurstReader::new(BufReader::new(file))
    }
}

impl<R: Read> BurstReader<R> {
    pub fn new(mut inner: R) -> io::Result<Self> {
        let mut header = [0u8; FILE_HEADER_LEN as usize];
        inner.read_exact(&mut header)?;
        if &header[..8] != FILE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a blue-dragon burst file",
            ));
        }
        if u16::from_le_bytes(header[8..10].try_into().unwrap()) != VERSION
            || u16::from_le_bytes(header[10..12].try_into().unwrap()) != FILE_HEADER_LEN
            || u32::from_le_bytes(header[12..16].try_into().unwrap()) != CHANNEL_SAMPLE_RATE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported burst file version",
            ));
        }
        Ok(Self { inner })
    }

    pub fn read_burst(&mut self) -> io::Result<Option<Burst>> {
        let mut header = [0u8; RECORD_HEADER_LEN as usize];
        match self.inner.read(&mut header[..1]) {
            Ok(0) => return Ok(None),
            Ok(1) => {}
            Ok(_) => unreachable!(),
            Err(e) => return Err(e),
        }
        self.inner.read_exact(&mut header[1..])?;
        if &header[..4] != RECORD_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid burst record marker",
            ));
        }

        let record_len = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let sample_count = u32::from_le_bytes(header[36..40].try_into().unwrap()) as usize;
        let expected_len = RECORD_HEADER_LEN as usize + sample_count.saturating_mul(4);
        if sample_count > MAX_SAMPLES_PER_RECORD || record_len as usize != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid burst record length",
            ));
        }
        let scale = f32::from_le_bytes(header[44..48].try_into().unwrap());
        if !scale.is_finite() || scale <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid IQ scale",
            ));
        }

        let mut packed = vec![0u8; sample_count * 4];
        self.inner.read_exact(&mut packed)?;
        let samples = packed
            .chunks_exact(4)
            .map(|pair| {
                let i = i16::from_le_bytes(pair[..2].try_into().unwrap()) as f32 * scale;
                let q = i16::from_le_bytes(pair[2..].try_into().unwrap()) as f32 * scale;
                Complex32::new(i, q)
            })
            .collect();

        let tv_nsec = u32::from_le_bytes(header[16..20].try_into().unwrap());
        if tv_nsec >= 1_000_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid burst timestamp",
            ));
        }

        Ok(Some(Burst {
            samples,
            freq: u32::from_le_bytes(header[20..24].try_into().unwrap()),
            num: u32::from_le_bytes(header[32..36].try_into().unwrap()),
            rssi_db: f32::from_le_bytes(header[24..28].try_into().unwrap()),
            noise_db: f32::from_le_bytes(header[28..32].try_into().unwrap()),
            timestamp: Timespec {
                tv_sec: u64::from_le_bytes(header[8..16].try_into().unwrap()),
                tv_nsec: tv_nsec as u64,
            },
            scan: u32::from_le_bytes(header[40..44].try_into().unwrap()) & FLAG_SCAN != 0,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn example_burst() -> Burst {
        Burst {
            samples: vec![Complex32::new(0.25, -0.5), Complex32::new(1.0, 0.0)],
            freq: 2441,
            num: 7,
            rssi_db: -42.5,
            noise_db: -91.0,
            timestamp: Timespec {
                tv_sec: 12,
                tv_nsec: 345,
            },
            scan: false,
        }
    }

    #[test]
    fn round_trips_channelized_iq_and_metadata() {
        let mut writer = BurstWriter::new(Vec::new(), 0).unwrap();
        assert!(writer.write_burst(&example_burst()).unwrap());
        let bytes = writer.into_inner();
        let mut reader = BurstReader::new(Cursor::new(bytes)).unwrap();
        let decoded = reader.read_burst().unwrap().unwrap();

        assert_eq!(decoded.freq, 2441);
        assert_eq!(decoded.num, 7);
        assert_eq!(decoded.timestamp.tv_nsec, 345);
        assert!((decoded.samples[0].re - 0.25).abs() < 0.0001);
        assert!((decoded.samples[0].im + 0.5).abs() < 0.0001);
        assert!(reader.read_burst().unwrap().is_none());
    }

    #[test]
    fn refuses_record_past_limit_without_partial_write() {
        let mut writer = BurstWriter::new(Vec::new(), 16 + 48 + 8 - 1).unwrap();
        assert!(!writer.write_burst(&example_burst()).unwrap());
        assert_eq!(writer.into_inner().len(), FILE_HEADER_LEN as usize);
    }

    #[test]
    fn rejects_truncated_record_header() {
        let mut bytes = Vec::from(FILE_MAGIC.as_slice());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&FILE_HEADER_LEN.to_le_bytes());
        bytes.extend_from_slice(&CHANNEL_SAMPLE_RATE.to_le_bytes());
        bytes.extend_from_slice(b"BR");
        let mut reader = BurstReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(
            reader.read_burst().unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }
}
