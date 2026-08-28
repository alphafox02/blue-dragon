// Copyright 2025-2026 CEMAXECUTER LLC

use num_complex::Complex32;
use crate::agc::{Agc, SquelchState};
use bd_protocol::Timespec;
use std::collections::VecDeque;

const BURST_START_SIZE: usize = 2048;
/// LE Coded S=8 with 251-byte payload = ~67K samples at SPS=2.
/// Must be large enough for the longest possible coded burst.
const MAX_BURST_SIZE: usize = 131072;
const BURST_RSSI_OFFSET: usize = 80;
/// Squelch timeout in samples.  After signal drops below threshold, keep
/// capturing for this many samples before ending the burst.  Matches the
/// C code's timeout of 100 samples.
const SQUELCH_TIMEOUT: u32 = 100;
// EDR continuation. An EDR packet keeps the GFSK access code and header, then
// switches to DQPSK/8DPSK, whose lower, non-constant envelope drops under the
// squelch and ends the burst partway through the payload. When the fixed EDR
// sync is present we extend the capture, bounded by the longest EDR packet, so
// the full payload reaches the demod and CRC. Classic (non-scan) channels only.
const MIN_EDR_BURST: usize = 400;         // must have captured past the sync
const MAX_EDR_HOLD: usize = 6400;         // ~5-slot 3-DHx at 2 Msps, hard bound
// The burst ends at the first envelope drop, so a cut EDR packet is short. Only
// bursts in this range are continuation candidates; longer BR and coded bursts
// are already complete and are left alone.
const EDR_TRUNC_MAX: usize = 1500;
const EDR_SYNC_MAX_SCORE: f32 = 0.15;     // strict; downstream CRC is authority
const EDR_PRE_ROLL: usize = 512;          // 256 us at 2 Msps
const CHANNEL_SAMPLE_PERIOD_NS: u64 = 500; // 2 Msps channelizer output

fn timestamp_with_sample_offset(start: &Timespec, sample_offset: usize) -> Timespec {
    let offset_ns = sample_offset as u64 * CHANNEL_SAMPLE_PERIOD_NS;
    let total_ns = start.tv_nsec + offset_ns;
    Timespec {
        tv_sec: start.tv_sec + total_ns / 1_000_000_000,
        tv_nsec: total_ns % 1_000_000_000,
    }
}

fn timestamp_before(end: &Timespec, sample_count: usize) -> Timespec {
    let end_ns = end.tv_sec as u128 * 1_000_000_000 + end.tv_nsec as u128;
    let offset_ns = sample_count.saturating_sub(1) as u128 * CHANNEL_SAMPLE_PERIOD_NS as u128;
    let start_ns = end_ns.saturating_sub(offset_ns);
    Timespec {
        tv_sec: (start_ns / 1_000_000_000) as u64,
        tv_nsec: (start_ns % 1_000_000_000) as u64,
    }
}

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
    /// True when the EDR gate selected this burst for a wideband IQ decode.
    pub edr_extended: bool,
    /// Number of leading pre-roll samples. Normal demodulation skips these;
    /// the EDR fallback uses them to recover a weak access code and header.
    pub edr_lead_samples: usize,
}

/// Per-channel burst catcher: feeds samples through AGC and detects burst boundaries
/// First-pass test before the more expensive sync correlation: EDR's DQPSK and
/// 8DPSK payloads have a non-constant envelope, while GFSK (both BR and BLE) is
/// constant-envelope after AGC. Comparing the envelope's coefficient of
/// variation lets constant-envelope bursts skip the sync search entirely. Only
/// runs at a Classic-channel timeout.
fn dpsk_envelope_signature(buf: &[Complex32]) -> bool {
    let n = buf.len();
    // Measure only the payload region: past the GFSK header (~270 samples in) and
    // before the squelch-drop tail (last ~120). The tail always varies as the
    // signal falls off, so including it would defeat the test. Across this window
    // GFSK stays flat while DPSK does not.
    let lo = 270usize;
    let hi = n.saturating_sub(120);
    if hi <= lo + 64 {
        return false;
    }
    let win = &buf[lo..hi];
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    for s in win {
        let a = s.norm();
        sum += a;
        sumsq += a * a;
    }
    let inv = 1.0 / win.len() as f32;
    let mean = sum * inv;
    if mean < 1e-6 {
        return false;
    }
    let var = (sumsq * inv - mean * mean).max(0.0);
    // GFSK sits around 0.05-0.12; DQPSK/8DPSK runs well above it.
    var.sqrt() / mean > 0.25
}

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
    /// Samples remaining to capture for an in-progress EDR continuation.
    edr_hold_remaining: usize,
    /// Continuation is opt-in with the wideband EDR path. BD_NO_EDR_CONT can
    /// still disable it independently for A/B testing.
    edr_enabled: bool,
    /// Recent samples retained while idle. EDR payload energy can cross
    /// squelch after its weaker GFSK access code and header have already passed.
    edr_pre_roll: VecDeque<Complex32>,
    edr_lead_in: Vec<Complex32>,
    edr_lead_samples: usize,
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
            edr_hold_remaining: 0,
            edr_enabled: std::env::var_os("BD_EDR_WIDEBAND").is_some()
                && std::env::var_os("BD_NO_EDR_CONT").is_none(),
            edr_pre_roll: VecDeque::with_capacity(EDR_PRE_ROLL),
            edr_lead_in: Vec::new(),
            edr_lead_samples: 0,
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
        self.execute_at(sample, now, 0)
    }

    /// Process a sample at an offset from a channelized batch timestamp.
    pub fn execute_at(
        &mut self,
        sample: Complex32,
        batch_start: &Timespec,
        sample_offset: usize,
    ) -> Option<Burst> {
        let (output, state) = self.agc.execute(sample);

        if self.edr_enabled && !self.capturing {
            if self.edr_pre_roll.len() == EDR_PRE_ROLL {
                self.edr_pre_roll.pop_front();
            }
            self.edr_pre_roll.push_back(output);
        }

        // Scan mode: accumulate every AGC-processed sample regardless of squelch
        if let Some(ref mut sbuf) = self.scan_buf {
            sbuf.push(output);
            self.scan_new += 1;
            if self.scan_new == 1 {
                self.scan_ts = timestamp_with_sample_offset(batch_start, sample_offset);
            }
        }

        // Continuation in progress: keep the rest of the DPSK payload, ignoring
        // the squelch state until the bounded sample count is reached.
        if self.edr_hold_remaining > 0 {
            if self.burst_buf.len() < MAX_BURST_SIZE {
                self.burst_buf.push(output);
            }
            self.edr_hold_remaining -= 1;
            if self.edr_hold_remaining == 0 {
                self.capturing = false;
                let burst = Burst {
                    samples: std::mem::take(&mut self.burst_buf),
                    freq: self.freq,
                    num: self.burst_num,
                    rssi_db: self.burst_rssi,
                    noise_db: self.agc.rssi_db(),
                    timestamp: self.timestamp.clone(),
                    scan: false,
                    edr_extended: true,
                    edr_lead_samples: std::mem::take(&mut self.edr_lead_samples),
                };
                self.burst_num += 1;
                return Some(burst);
            }
            return None;
        }

        match state {
            SquelchState::Rise => {
                // Start of a new burst
                self.burst_buf = Vec::with_capacity(BURST_START_SIZE);
                self.burst_rssi = -127.0;
                let rise = timestamp_with_sample_offset(batch_start, sample_offset);
                if self.edr_enabled {
                    self.edr_lead_in.clear();
                    self.edr_lead_in.extend(self.edr_pre_roll.drain(..));
                }
                self.timestamp = rise;
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
                    // If this Classic burst carries the fixed EDR sync it was
                    // cut off partway through the payload. Extend the capture
                    // instead of emitting the fragment. GFSK bursts and noise do
                    // not match the sync, so BLE and BR are unaffected.
                    if self.edr_enabled
                        && self.scan_buf.is_none()
                        && self.burst_buf.len() >= MIN_EDR_BURST
                        && self.burst_buf.len() < EDR_TRUNC_MAX
                        && dpsk_envelope_signature(&self.burst_buf)
                        && crate::edr::detect_edr_sync(&self.burst_buf, 2, EDR_SYNC_MAX_SCORE)
                            .is_some()
                    {
                        let lead = std::mem::take(&mut self.edr_lead_in);
                        if !lead.is_empty() {
                            self.edr_lead_samples = lead.len();
                            self.timestamp = timestamp_before(&self.timestamp, lead.len());
                            let burst = std::mem::take(&mut self.burst_buf);
                            self.burst_buf = Vec::with_capacity(MAX_EDR_HOLD);
                            self.burst_buf.extend(lead);
                            self.burst_buf.extend(burst);
                        }
                        self.edr_hold_remaining = MAX_EDR_HOLD - self.burst_buf.len();
                        return None;
                    }
                    self.capturing = false;
                    let edr_candidate = self.edr_enabled
                        && self.scan_buf.is_none()
                        && self.burst_buf.len() >= MIN_EDR_BURST
                        && dpsk_envelope_signature(&self.burst_buf);
                    if edr_candidate {
                        let lead = std::mem::take(&mut self.edr_lead_in);
                        let edr_lead_samples = lead.len();
                        if std::env::var_os("BD_EDR_DEBUG").is_some() {
                            eprintln!(
                                "[edr-catcher] freq={} burst={} lead={}",
                                self.freq,
                                self.burst_buf.len(),
                                edr_lead_samples
                            );
                        }
                        if !lead.is_empty() {
                            self.timestamp = timestamp_before(&self.timestamp, lead.len());
                            let burst = std::mem::take(&mut self.burst_buf);
                            self.burst_buf = Vec::with_capacity(lead.len() + burst.len());
                            self.burst_buf.extend(lead);
                            self.burst_buf.extend(burst);
                        }
                        let burst = Burst {
                            samples: std::mem::take(&mut self.burst_buf),
                            freq: self.freq,
                            num: self.burst_num,
                            rssi_db: self.burst_rssi,
                            noise_db: self.agc.rssi_db(),
                            timestamp: self.timestamp.clone(),
                            scan: false,
                            edr_extended: true,
                            edr_lead_samples,
                        };
                        self.burst_num += 1;
                        return Some(burst);
                    } else {
                        self.edr_lead_in.clear();
                    }
                    let burst = Burst {
                        samples: std::mem::take(&mut self.burst_buf),
                        freq: self.freq,
                        num: self.burst_num,
                        rssi_db: self.burst_rssi,
                        noise_db: self.agc.rssi_db(),
                        timestamp: self.timestamp.clone(),
                        scan: false,
                        edr_extended: false,
                        edr_lead_samples: 0,
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
            edr_extended: false,
            edr_lead_samples: 0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_offset_timestamp_crosses_second_boundary() {
        let start = Timespec {
            tv_sec: 10,
            tv_nsec: 999_999_000,
        };
        let result = timestamp_with_sample_offset(&start, 4);
        assert_eq!(result.tv_sec, 11);
        assert_eq!(result.tv_nsec, 1_000);
    }
}
