// Copyright 2025-2026 CEMAXECUTER LLC

pub mod file;
pub mod vita49;

#[cfg(feature = "usrp")]
pub mod usrp;

#[cfg(feature = "hackrf")]
pub mod hackrf;

#[cfg(feature = "bladerf")]
pub mod bladerf;

#[cfg(feature = "soapysdr")]
pub mod soapysdr;

#[cfg(feature = "aaronia")]
pub mod aaronia;

#[cfg(feature = "rfnm")]
pub mod rfnm;

use crossbeam::channel::Sender;

/// IQ sample format (shared across file, VITA 49, and other backends)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IqFormat {
    /// Complex int8 (CS8): pairs of i8
    Ci8,
    /// Complex int16 (CS16): pairs of i16
    Ci16,
    /// Complex float32 (CF32): pairs of f32
    Cf32,
}

/// IQ format conversion utilities.
///
/// All functions write into a caller-provided `dst` slice and return the number
/// of i16 values written. No heap allocation on the hot path.
pub mod convert {
    /// Convert CI8 (signed 8-bit) to i16, scaling by <<8.
    pub fn ci8_to_i16(src: &[u8], dst: &mut [i16]) -> usize {
        let n = src.len().min(dst.len());
        for i in 0..n {
            dst[i] = (src[i] as i8 as i16) << 8;
        }
        n
    }

    /// Convert big-endian CI16 to native i16.
    pub fn ci16_be_to_i16(src: &[u8], dst: &mut [i16]) -> usize {
        let max = dst.len().min(src.len() / 2);
        for (i, chunk) in src.chunks_exact(2).take(max).enumerate() {
            dst[i] = i16::from_be_bytes([chunk[0], chunk[1]]);
        }
        max
    }

    /// Convert little-endian CI16 to native i16.
    pub fn ci16_le_to_i16(src: &[u8], dst: &mut [i16]) -> usize {
        let max = dst.len().min(src.len() / 2);
        for (i, chunk) in src.chunks_exact(2).take(max).enumerate() {
            dst[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
        }
        max
    }

    /// Convert big-endian CF32 to i16 (scale [-1,1] -> [-32768,32767]).
    pub fn cf32_be_to_i16(src: &[u8], dst: &mut [i16]) -> usize {
        let max = dst.len().min(src.len() / 4);
        for (i, chunk) in src.chunks_exact(4).take(max).enumerate() {
            let f = f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            dst[i] = (f * 32767.0).clamp(-32768.0, 32767.0) as i16;
        }
        max
    }

    /// Convert little-endian CF32 to i16 (scale [-1,1] -> [-32768,32767]).
    pub fn cf32_le_to_i16(src: &[u8], dst: &mut [i16]) -> usize {
        let max = dst.len().min(src.len() / 4);
        for (i, chunk) in src.chunks_exact(4).take(max).enumerate() {
            let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            dst[i] = (f * 32767.0).clamp(-32768.0, 32767.0) as i16;
        }
        max
    }
}

/// Sample buffer: a block of interleaved int16 IQ samples
pub struct SampleBuf {
    /// Interleaved I,Q,I,Q,... as i16
    pub data: Vec<i16>,
    /// Number of complex samples (data.len() / 2)
    pub num_samples: usize,
}

/// Common trait for all SDR backends
pub trait SdrSource: Send {
    /// Start streaming samples into the channel.
    /// This function should run until stop() is called or an error occurs.
    fn start(&mut self, tx: Sender<SampleBuf>) -> Result<(), String>;

    /// Signal the source to stop streaming
    fn stop(&mut self);

    /// Get the sample rate in Hz
    fn sample_rate(&self) -> u32;

    /// Get the center frequency in Hz
    fn center_frequency(&self) -> u64;
}
