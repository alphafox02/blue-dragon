use num_complex::Complex32;
use crate::agc::{Agc, SquelchState};
use bd_protocol::Timespec;

const BURST_START_SIZE: usize = 2048;
/// LE Coded S=8 with 251-byte payload = ~67K samples at SPS=2.
/// Must be large enough for the longest possible coded burst.
const MAX_BURST_SIZE: usize = 131072;
const BURST_RSSI_OFFSET: usize = 80;
/// Squelch timeout in samples.  After signal drops below threshold, keep
/// capturing for this many samples before ending the burst.  Matches the
/// C code's timeout of 100 samples.
const SQUELCH_TIMEOUT: u32 = 100;

/// Scan window size for advertising channel continuous capture.
/// Must fit at least one coded S=8 burst (~6560 samples for DRI-sized PDU)
/// plus the step size overlap.  At SPS=2, 49152 samples = 24.6 ms.
const SCAN_WINDOW: usize = 49152;
/// New samples between scan burst emissions.  32768 samples = 16.4 ms at
/// 2 Msps.  With 3 advertising channels, this produces ~183 scan bursts/s
/// (vs 732/s at 8192), keeping decode thread load under 10%.
/// Overlap = SCAN_WINDOW - SCAN_STEP = 16384 samples (8 ms), enough for
/// any coded packet to fit entirely within at least one window.
const SCAN_STEP: usize = 32768;

/// A detected burst of IQ samples
#[derive(Debug)]
pub struct Burst {
    pub samples: Vec<Complex32>,
    pub freq: u32,
    pub num: u32,
    pub rssi_db: f32,
    pub noise_db: f32,
    pub timestamp: Timespec,
    /// True if this is a scan burst (continuous capture, not squelch-triggered).
    /// Scan bursts should only be processed for coded PHY decode.
    pub scan: bool,
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
    /// Scan mode: continuously capture samples for coded PHY search on
    /// advertising channels, regardless of squelch.
    scan_buf: Option<Vec<Complex32>>,
    scan_new: usize,
    scan_ts: Timespec,
}

impl BurstCatcher {
    pub fn new(freq: u32, squelch_db: f32) -> Self {
        let agc = Agc::new(0.25, squelch_db, SQUELCH_TIMEOUT);
        Self {
            freq,
            agc,
            burst_buf: Vec::new(),
            burst_rssi: -127.0,
            burst_num: 0,
            timestamp: Timespec::default(),
            capturing: false,
            scan_buf: None,
            scan_new: 0,
            scan_ts: Timespec::default(),
        }
    }

    /// Create a scan-enabled burst catcher for advertising channels.
    /// In addition to normal squelch-based burst detection, this continuously
    /// captures AGC-processed samples and periodically emits scan bursts
    /// for coded PHY preamble search.
    pub fn new_scan(freq: u32, squelch_db: f32) -> Self {
        let mut bc = Self::new(freq, squelch_db);
        bc.scan_buf = Some(Vec::with_capacity(SCAN_WINDOW));
        bc
    }

    /// Process one IQ sample. Returns Some(Burst) when a complete burst has been detected
    /// (signal rose above squelch, accumulated samples, then fell back below and timed out).
    pub fn execute(&mut self, sample: Complex32, now: &Timespec) -> Option<Burst> {
        let (output, state) = self.agc.execute(sample);

        // Scan mode: accumulate every AGC-processed sample regardless of squelch
        if let Some(ref mut sbuf) = self.scan_buf {
            sbuf.push(output);
            self.scan_new += 1;
            if self.scan_new == 1 {
                self.scan_ts = now.clone();
            }
        }

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
                        scan: false,
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

    /// Check if a scan burst is ready.  Returns Some(Burst) with scan=true
    /// when enough new samples have accumulated since the last scan emission.
    pub fn take_scan_burst(&mut self) -> Option<Burst> {
        if self.scan_new < SCAN_STEP {
            return None;
        }
        let sbuf = self.scan_buf.as_mut()?;
        if sbuf.len() < SCAN_STEP {
            return None;
        }

        let burst = Burst {
            samples: sbuf.clone(),
            freq: self.freq,
            num: self.burst_num,
            rssi_db: self.agc.rssi_db(),
            noise_db: -127.0,
            timestamp: self.scan_ts.clone(),
            scan: true,
        };
        self.burst_num += 1;

        // Keep overlap: drain the oldest samples, keep SCAN_WINDOW - SCAN_STEP
        let keep = SCAN_WINDOW.saturating_sub(SCAN_STEP);
        if sbuf.len() > keep {
            let drain_count = sbuf.len() - keep;
            sbuf.drain(..drain_count);
        }
        self.scan_new = 0;

        Some(burst)
    }

    /// Set squelch threshold
    pub fn set_squelch(&mut self, threshold_db: f32) {
        self.agc.set_squelch_threshold(threshold_db);
    }
}
