use num_complex::Complex32;
use crate::agc::{Agc, SquelchState};
use bd_protocol::Timespec;

const BURST_START_SIZE: usize = 2048;
const MAX_BURST_SIZE: usize = 32768;
const BURST_RSSI_OFFSET: usize = 80;

/// A detected burst of IQ samples
#[derive(Debug)]
pub struct Burst {
    pub samples: Vec<Complex32>,
    pub freq: u32,
    pub num: u32,
    pub rssi_db: f32,
    pub noise_db: f32,
    pub timestamp: Timespec,
}

/// Per-channel burst catcher: feeds samples through AGC and detects burst boundaries
pub struct BurstCatcher {
    freq: u32,
    agc: Agc,
    burst_buf: Vec<Complex32>,
    burst_rssi: f32,
    burst_num: u32,
    timestamp: Timespec,
    capturing: bool,
}

impl BurstCatcher {
    pub fn new(freq: u32, squelch_db: f32) -> Self {
        Self {
            freq,
            agc: Agc::new(0.25, squelch_db, 100),
            burst_buf: Vec::new(),
            burst_rssi: -127.0,
            burst_num: 0,
            timestamp: Timespec::default(),
            capturing: false,
        }
    }

    /// Process one IQ sample. Returns Some(Burst) when a complete burst has been detected
    /// (signal rose above squelch, accumulated samples, then fell back below and timed out).
    pub fn execute(&mut self, sample: Complex32, now: &Timespec) -> Option<Burst> {
        let (output, state) = self.agc.execute(sample);

        match state {
            SquelchState::Rise => {
                // Start of a new burst
                self.burst_buf = Vec::with_capacity(BURST_START_SIZE);
                self.burst_rssi = -127.0;
                self.timestamp = now.clone();
                self.capturing = true;
                None
            }
            SquelchState::SignalHi => {
                if self.capturing && self.burst_buf.len() < MAX_BURST_SIZE {
                    self.burst_buf.push(output);
                    if self.burst_buf.len() == BURST_RSSI_OFFSET {
                        self.burst_rssi = self.agc.rssi_db();
                    }
                }
                None
            }
            SquelchState::Timeout => {
                if self.capturing && !self.burst_buf.is_empty() {
                    self.capturing = false;
                    let burst = Burst {
                        samples: std::mem::take(&mut self.burst_buf),
                        freq: self.freq,
                        num: self.burst_num,
                        rssi_db: self.burst_rssi,
                        noise_db: self.agc.rssi_db(),
                        timestamp: self.timestamp.clone(),
                    };
                    self.burst_num += 1;
                    Some(burst)
                } else {
                    self.capturing = false;
                    None
                }
            }
            SquelchState::SignalLo => {
                None
            }
        }
    }

    /// Set squelch threshold
    pub fn set_squelch(&mut self, threshold_db: f32) {
        self.agc.set_squelch_threshold(threshold_db);
    }
}
