// Copyright 2025-2026 CEMAXECUTER LLC

pub mod file;

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

use crossbeam::channel::Sender;

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
