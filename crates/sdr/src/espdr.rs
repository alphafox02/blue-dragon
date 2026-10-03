// Copyright 2025-2026 CEMAXECUTER LLC

//! ESP32-S3 receiver using the eSpDR radio firmware, over the ESP's own
//! USB Serial/JTAG port (no FPGA link).
//!
//! The ESP must run the eSpDR firmware with the USB modes of the
//! alphafox02/eSpDR fork, loaded into RAM with `esptool load-ram`. USB Full
//! Speed carries far less than the radio produces, so the firmware sends
//! only part of it, in one of two ways:
//!
//! * stream (preferred): the ESP watches a 16 MHz window continuously and
//!   sends each burst above the noise floor, timestamped in samples. The
//!   backend places bursts on a true timeline and fills the gaps with noise
//!   at the reported floor. The ESP cuts each burst down to its own channel
//!   at 4 Msps (about 4x less data over USB, so more bursts get through and
//!   long ones such as 3-DH5 fit), and the backend restores it to the 16 MHz
//!   window; `BD_ESPDR_WIDE` asks for whole-window bursts instead, which keep
//!   simultaneous signals on different channels. Wi-Fi bursts are dropped on
//!   the ESP unless `BD_ESPDR_KEEP_WIDEBAND` is set.
//! * snapshot (fallback, for firmware without streaming, at 80 Msps, or with
//!   `BD_ESPDR_SNAPSHOT` set): the
//!   ESP sends about 1 ms of samples per request, and snapshots are joined
//!   with short noise gaps, so sample time runs faster than wall-clock time.
//!
//! Either way, eSpDR's I + jQ uses the LO-minus-RF convention, so samples
//! are conjugated to the usual RF-minus-LO orientation, and the 10-bit
//! values are scaled by 64 toward int16 full scale.
//!
//! Interface strings: `espdr` or `espdr0` (first ESP32-S3 USB serial port),
//! `espdrN` (the Nth, sorted by serial path), or `espdr:/dev/ttyACM0`.
//!
//! An ESP at 16 Msps hears about 80 MHz around its LO folded into its 16 MHz
//! window (the 80 Msps capture is decimated without filtering), so `-C 80`
//! with `-i espdr` (every attached ESP) or a comma list (`espdr0,espdr2`)
//! receives the whole folded band and lets the decoders sort out where each
//! burst came from (see `multi`).

mod multi;

use crossbeam::channel::{bounded, Receiver, Sender, TrySendError};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const REQ_MAGIC: u8 = 0xB4;
const RSP_MAGIC: u8 = 0xB5;
const CTL_INFO: u8 = 1;
const CTL_STATUS: u8 = 3;
const ESP_ARG_HIGH: u8 = 19;
const ESP_SET_LO: u8 = 20;
const ESP_SET_RATE: u8 = 21;
const ESP_SET_WIDTH: u8 = 22;
const ESP_SET_FILTER: u8 = 23;
const ESP_SET_GAIN: u8 = 24;
const ESP_SNAPSHOT: u8 = 40;
const ESP_STREAM: u8 = 41;
const ESP_STAT_RADIO: u16 = 13;
const CTL_UNKNOWN_OP: u8 = 1;
const CTL_NOT_READY: u8 = 4;
const CTL_ESP_FIRMWARE_ID: u32 = 0x4951_5305;
const SNAP_MAGIC: u32 = 0x5041_4E53;
const STREAM_MAGIC: u32 = 0x5453_5242;
const STREAM_BURST: u16 = 1;
const STREAM_STATUS: u16 = 2;
const STREAM_END: u16 = 3;
const STREAM_NARROW: u16 = 4;
const STREAM_STATUS_V2: u16 = 5;
const STREAM_REJECT_WIDEBAND: u16 = 1;
const STREAM_CHANNELIZE: u16 = 2;
const STREAM_TELEMETRY: u16 = 4;
const STREAM_TRIGGER_RATIO_SHIFT: u16 = 3;
const STREAM_STATUS_USB_SOF: u32 = 0x8000;
const STREAM_STATUS_USB_FRAME: u32 = 0x07FF;
const STATUS_WORDS: usize = 8;
const STATUS_V2_WORDS: usize = 16;
/// Largest stretch of noise inserted for one gap (0.1 s at 16 Msps); a longer
/// gap only happens across a restart, and its time is skipped.
const MAX_GAP_PAIRS: u64 = 1_600_000;
/// Pairs per chunk handed to the pipeline.
const CHUNK_PAIRS: usize = 32768;

/// Noise-filled pairs placed between snapshots.
const GAP_PAIRS: usize = 2048;
/// Scale from 10-bit samples toward int16 full scale.
const SAMPLE_SCALE: i32 = 64;
/// Baseband filter capacitor code for 16 Msps (0..63; larger is narrower).
/// The 16 Msps samples are the 80 Msps capture decimated by five without a
/// digital filter, so this analog filter is the only thing keeping signals
/// from outside the window from folding into it: at the firmware's default
/// (0, about 69 MHz) a signal 23 MHz away came through as strongly as one in
/// the window. Measured on channel 39: at 54 one 9 or 23 MHz outside the
/// window no longer came through, while one 7 MHz inside still did, a dB or
/// so down. (Codes from the ESP-SDR project's S3 calibration: 48 is about
/// 16 MHz wide, 60 about 13.)
const FILTER_16M: u16 = 54;
/// The ESP's channel filter (Q14), also used here to interpolate back to
/// 16 Msps. Narrow output j is centred on pair start + 4j - 2.5.
const NARROW_TAPS: [i32; 12] = [-27, 62, 476, 1428, 2676, 3577, 3577, 2676, 1428, 476, 62, -27];

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// Request/response link to the ESP firmware's control protocol.
struct EspLink {
    port: Box<dyn serialport::SerialPort>,
    sequence: u16,
}

impl EspLink {
    fn open(path: &str) -> Result<Self, String> {
        let port = serialport::new(path, 115_200)
            .timeout(Duration::from_secs(2))
            .open()
            .map_err(|e| format!("eSpDR: cannot open {}: {}", path, e))?;
        let mut link = Self { port, sequence: 0 };
        link.drain();
        Ok(link)
    }

    /// Discards input until the line has been quiet for 300 ms (at most 5 s).
    /// A previous host may have stopped reading in the middle of a snapshot,
    /// leaving the ESP still sending it.
    fn drain(&mut self) {
        let _ = self.port.set_timeout(Duration::from_millis(300));
        let start = std::time::Instant::now();
        let mut scratch = [0u8; 4096];
        while start.elapsed() < Duration::from_secs(5) {
            match self.port.read(&mut scratch) {
                Ok(n) if n > 0 => continue,
                _ => break,
            }
        }
        let _ = self.port.clear(serialport::ClearBuffer::Input);
        let _ = self.port.set_timeout(Duration::from_secs(2));
    }

    /// Identifies the firmware, resynchronising up to three times.
    fn handshake(&mut self) -> Result<u32, String> {
        let mut last = String::new();
        for _ in 0..3 {
            match self.command(CTL_INFO, 0) {
                Ok(id) => return Ok(id),
                Err(e) => {
                    last = e;
                    self.drain();
                }
            }
        }
        Err(last)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), String> {
        self.port
            .read_exact(buf)
            .map_err(|e| format!("eSpDR: serial read failed: {}", e))
    }

    /// Sends one request and returns (status, value).
    fn request(&mut self, op: u8, arg: u16) -> Result<(u8, u32), String> {
        self.sequence = self.sequence.wrapping_add(1);
        let mut req = [0u8; 10];
        req[0] = REQ_MAGIC;
        req[1] = op;
        req[2..4].copy_from_slice(&arg.to_le_bytes());
        req[4..6].copy_from_slice(&self.sequence.to_le_bytes());
        let crc = crc32(&req[..6]);
        req[6..10].copy_from_slice(&crc.to_le_bytes());
        self.port
            .write_all(&req)
            .map_err(|e| format!("eSpDR: serial write failed: {}", e))?;

        let mut rsp = [0u8; 16];
        self.read_exact(&mut rsp)?;
        let seq = u16::from_le_bytes([rsp[4], rsp[5]]);
        let value = u32::from_le_bytes([rsp[8], rsp[9], rsp[10], rsp[11]]);
        let crc = u32::from_le_bytes([rsp[12], rsp[13], rsp[14], rsp[15]]);
        if rsp[0] != RSP_MAGIC || rsp[2] != op || seq != self.sequence || crc != crc32(&rsp[..12]) {
            return Err(format!("eSpDR: malformed reply to op {}", op));
        }
        Ok((rsp[3], value))
    }

    fn command(&mut self, op: u8, arg: u16) -> Result<u32, String> {
        match self.request(op, arg)? {
            (0, value) => Ok(value),
            (status, _) => Err(format!("eSpDR: op {} failed with status {}", op, status)),
        }
    }

    fn command32(&mut self, op: u8, arg: u32) -> Result<u32, String> {
        self.command(ESP_ARG_HIGH, (arg >> 16) as u16)?;
        self.command(op, arg as u16)
    }

    /// Takes one snapshot and returns its raw 32-bit dump words.
    fn snapshot(&mut self) -> Result<Vec<u32>, String> {
        let (status, pairs) = self.request(ESP_SNAPSHOT, 0)?;
        if status == CTL_UNKNOWN_OP {
            return Err("eSpDR: firmware has no snapshot command; load the eSpDR \
                        snapshot firmware into RAM first"
                .to_string());
        }
        if status != 0 {
            return Err(format!("eSpDR: snapshot failed with status {}", status));
        }
        let mut header = [0u8; 16];
        self.read_exact(&mut header)?;
        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let count = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if magic != SNAP_MAGIC || count != pairs {
            return Err("eSpDR: bad snapshot header".to_string());
        }
        let mut raw = vec![0u8; count as usize * 4];
        self.read_exact(&mut raw)?;
        let mut crc = [0u8; 4];
        self.read_exact(&mut crc)?;
        if u32::from_le_bytes(crc) != crc32(&raw) {
            return Err("eSpDR: snapshot CRC mismatch".to_string());
        }
        Ok(raw
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
    }
}

/// One record of the ESP's burst stream.
enum Record {
    Burst { start: u64, pairs: Vec<u32> },
    /// One channel of a burst at 4 Msps, mixed down by `offset` MHz
    /// (LO-minus-RF orientation, like the pairs).
    Narrow { start: u64, offset: i32, pairs: Vec<u32> },
    Status { start: u64, usb_frame: Option<u16>, words: Vec<u32> },
    End,
}

impl EspLink {
    /// Starts burst streaming. Ok(false) if the firmware cannot stream (no
    /// command, or not at this sample rate).
    fn start_stream(&mut self, arg: u16) -> Result<bool, String> {
        self.start_stream_masked(arg, 0)
    }

    /// Starts burst streaming of only the channel positions in `mask` (bit
    /// k + 8 for offset k; 0 for all), passed as the argument's high half.
    fn start_stream_masked(&mut self, arg: u16, mask: u16) -> Result<bool, String> {
        if mask != 0 {
            self.command(ESP_ARG_HIGH, mask)?;
        }
        match self.request(ESP_STREAM, arg)? {
            (0, _) => Ok(true),
            (CTL_UNKNOWN_OP, _) | (CTL_NOT_READY, _) => Ok(false),
            (status, _) => Err(format!("eSpDR: stream failed with status {}", status)),
        }
    }

    /// Reads the next stream record, resynchronising on the magic if needed.
    fn next_record(&mut self) -> Result<Record, String> {
        let mut window = [0u8; 4];
        self.read_exact(&mut window)?;
        while u32::from_le_bytes(window) != STREAM_MAGIC {
            window.copy_within(1.., 0);
            self.read_exact(&mut window[3..])?;
        }
        let mut header = [0u8; 20];
        self.read_exact(&mut header)?;
        let word = |i: usize| u32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
        let kind = (word(0) & 0xFFFF) as u16;
        let flags = word(0) >> 16;
        let start = word(8) as u64 | (word(12) as u64) << 32;
        let length = word(16) as usize;
        let mut check = [0u8; 4];
        match kind {
            STREAM_BURST | STREAM_NARROW => {
                let mut payload = vec![0u8; length.div_ceil(2) * 5];
                self.read_exact(&mut payload)?;
                self.read_exact(&mut check)?;
                let mut pairs = Vec::with_capacity(length + 1);
                for g in payload.chunks_exact(5) {
                    let lo = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
                    pairs.push(lo & 0xF_FFFF);
                    pairs.push(lo >> 20 | (g[4] as u32) << 12);
                }
                let sum = pairs.iter().fold(0u32, |a, &p| a.wrapping_add(p));
                if sum != u32::from_le_bytes(check) {
                    return Err("eSpDR: stream burst check failed".to_string());
                }
                pairs.truncate(length);
                if kind == STREAM_NARROW {
                    let offset = ((flags >> 8) & 15) as i32 - 8;
                    Ok(Record::Narrow { start, offset, pairs })
                } else {
                    Ok(Record::Burst { start, pairs })
                }
            }
            STREAM_STATUS | STREAM_STATUS_V2 => {
                let count = if kind == STREAM_STATUS { STATUS_WORDS } else { STATUS_V2_WORDS };
                if kind == STREAM_STATUS_V2 && length != STATUS_V2_WORDS {
                    return Err(format!("eSpDR: bad extended status length {}", length));
                }
                let mut payload = vec![0u8; count * 4];
                self.read_exact(&mut payload)?;
                self.read_exact(&mut check)?;
                let words: Vec<u32> = payload
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let sum = words.iter().fold(0u32, |a, &w| a.wrapping_add(w));
                if sum != u32::from_le_bytes(check) {
                    return Err("eSpDR: stream status check failed".to_string());
                }
                let usb_frame = (flags & STREAM_STATUS_USB_SOF != 0)
                    .then_some((flags & STREAM_STATUS_USB_FRAME) as u16);
                Ok(Record::Status { start, usb_frame, words })
            }
            STREAM_END => {
                self.read_exact(&mut check)?;
                Ok(Record::End)
            }
            _ => Err(format!("eSpDR: unknown stream record type {}", kind)),
        }
    }

    /// Stops the stream and reads up to its END record.
    fn stop_stream(&mut self) {
        let _ = self.port.write_all(&[0]);
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while std::time::Instant::now() < deadline {
            match self.next_record() {
                Ok(Record::End) => return,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        self.drain();
    }
}

/// Places stream bursts on a continuous timeline, filling the gaps with noise
/// at the reported floor, and hands the result to the pipeline in chunks.
struct Timeline {
    tx: Sender<Vec<i16>>,
    /// Output sample index of the current stream's pair 0.
    base: u64,
    /// Output samples produced so far.
    emitted: u64,
    /// Noise standard deviation per component, in output units.
    sigma: f32,
    noise: Vec<f32>,
    noise_pos: usize,
}

impl Timeline {
    fn new(tx: Sender<Vec<i16>>) -> Self {
        // Unit-variance Gaussian noise (Box-Muller), reused cyclically.
        let mut state = 0x9E37_79B9u32;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 + 1.0) / (u32::MAX as f32 + 2.0)
        };
        let mut noise = Vec::with_capacity(1 << 16);
        while noise.len() < 1 << 16 {
            let r = (-2.0 * uniform().ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * uniform();
            noise.push(r * theta.cos());
            noise.push(r * theta.sin());
        }
        Self { tx, base: 0, emitted: 0, sigma: 4.0 * SAMPLE_SCALE as f32, noise, noise_pos: 0 }
    }

    fn send(&self, chunk: Vec<i16>) -> bool {
        self.tx.send(chunk).is_ok()
    }

    /// Fills with noise up to output index `target`. False once the pipeline
    /// has gone away.
    fn fill_to(&mut self, target: u64) -> bool {
        if target > self.emitted + MAX_GAP_PAIRS {
            self.emitted = target - MAX_GAP_PAIRS;
        }
        while self.emitted < target {
            let n = ((target - self.emitted) as usize).min(CHUNK_PAIRS);
            let mut chunk = Vec::with_capacity(2 * n);
            for _ in 0..2 * n {
                chunk.push((self.noise[self.noise_pos] * self.sigma) as i16);
                self.noise_pos = (self.noise_pos + 1) & (self.noise.len() - 1);
            }
            self.emitted += n as u64;
            if !self.send(chunk) {
                return false;
            }
        }
        true
    }

    /// Adds a burst, as interleaved output samples, that starts at stream
    /// pair `start`.
    fn burst(&mut self, start: u64, samples: &[i16]) -> bool {
        let at = self.base + start;
        if !self.fill_to(at) {
            return false;
        }
        let pairs = samples.len() / 2;
        let skip = (self.emitted.saturating_sub(at) as usize).min(pairs);
        for piece in samples[2 * skip..].chunks(2 * CHUNK_PAIRS) {
            if !self.send(piece.to_vec()) {
                return false;
            }
        }
        self.emitted = self.emitted.max(at + pairs as u64);
        true
    }

    /// Continues the timeline across a stream restart.
    fn restart(&mut self) {
        self.base = self.emitted;
    }
}

/// Converts 20-bit dump pairs to conjugated, scaled int16 I/Q.
fn convert_pairs(pairs: &[u32]) -> Vec<i16> {
    let mut out = Vec::with_capacity(2 * pairs.len());
    for &w in pairs {
        let i = ((w & 1023) ^ 512) as i32 - 512;
        let q = (((w >> 10) & 1023) ^ 512) as i32 - 512;
        out.push((i * SAMPLE_SCALE).clamp(-32768, 32767) as i16);
        out.push((-q * SAMPLE_SCALE).clamp(-32768, 32767) as i16);
    }
    out
}

/// Removes the board's DC offset from a narrow record (I, Q as sent, mixed
/// down by `offset` MHz). The ESP's DC offset is often 20 dB above the noise
/// in a 1 MHz channel, which ruins weak packets on the two channels either
/// side of the LO. In the record it is a tone at -`offset` MHz, a quarter
/// turn per output sample for each MHz, so it is measured over the record
/// and subtracted: a notch a few kHz wide at a frequency no Bluetooth channel
/// uses when tiled (the LOs sit half a MHz off the channels). Only tiled
/// boards use it: a single board or folded boards can have a channel on the
/// LO, where the notch would take part of the packet.
fn remove_dc(y: &mut [(f32, f32)], offset: i32) {
    if offset.abs() > 2 || y.len() < 64 {
        return; // outside the record's +-2 MHz, or too short to measure
    }
    // e^(-j pi offset n / 2): cycles through 1, -j, -1, j for offset 1.
    let turn = |n: usize| match ((-(offset as i64) * n as i64).rem_euclid(4)) as u8 {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        2 => (-1.0, 0.0),
        _ => (0.0, -1.0),
    };
    let (mut si, mut sq) = (0.0f64, 0.0f64);
    for (n, &(i, q)) in y.iter().enumerate() {
        let (c, s) = turn(n);
        // y times the conjugate of the tone.
        si += (i * c + q * s) as f64;
        sq += (q * c - i * s) as f64;
    }
    let (di, dq) = ((si / y.len() as f64) as f32, (sq / y.len() as f64) as f32);
    for (n, v) in y.iter_mut().enumerate() {
        let (c, s) = turn(n);
        v.0 -= di * c - dq * s;
        v.1 -= di * s + dq * c;
    }
}

/// Restores a narrow record, starting at stream pair `start`, to the 16 MHz
/// window: interpolates by 4 with the ESP's filter, mixes back up by
/// `offset` MHz, then conjugates and scales like `convert_pairs`.
fn convert_narrow(pairs: &[u32], start: u64, offset: i32) -> Vec<i16> {
    let unit = |m: i64| {
        let a = 2.0 * std::f32::consts::PI * (m.rem_euclid(16) as f32) / 16.0;
        (a.cos(), a.sin())
    };
    let y: Vec<(f32, f32)> = pairs
        .iter()
        .map(|&w| (((w & 1023) ^ 512) as f32 - 512.0, (((w >> 10) & 1023) ^ 512) as f32 - 512.0))
        .collect();
    let gain = 4.0 / 16384.0;
    let mut out = Vec::with_capacity(8 * y.len());
    for i in 0..4 * y.len() {
        // Output i of the full interpolating convolution is centred on pair
        // start - 8 + i; take pairs from start on.
        let c = i + 8;
        let (mut zi, mut zq) = (0.0f32, 0.0f32);
        let mut t = c % 4;
        while t < NARROW_TAPS.len() {
            let j = (c - t) / 4;
            if j < y.len() {
                zi += y[j].0 * NARROW_TAPS[t] as f32;
                zq += y[j].1 * NARROW_TAPS[t] as f32;
            }
            t += 4;
        }
        let (cos, sin) = unit(offset as i64 * (start + i as u64) as i64);
        let (ri, rq) = ((zi * cos - zq * sin) * gain, (zi * sin + zq * cos) * gain);
        let scale = SAMPLE_SCALE as f32;
        out.push((ri.round() * scale).clamp(-32768.0, 32767.0) as i16);
        out.push((-rq.round() * scale).clamp(-32768.0, 32767.0) as i16);
    }
    out
}

/// Converts dump words to conjugated, scaled int16 pairs, followed by a
/// noise-filled gap at the snapshot's measured noise floor.
fn convert_snapshot(words: &[u32], rng: &mut u32) -> Vec<i16> {
    let mut out = Vec::with_capacity(2 * (words.len() + GAP_PAIRS));
    for &w in words {
        let i = ((w & 1023) ^ 512) as i32 - 512;
        let q = (((w >> 10) & 1023) ^ 512) as i32 - 512;
        out.push((i * SAMPLE_SCALE).clamp(-32768, 32767) as i16);
        out.push((-q * SAMPLE_SCALE).clamp(-32768, 32767) as i16);
    }

    // Noise floor: 10th-percentile power of 256-pair blocks.
    let mut block_power: Vec<f32> = out
        .chunks(512)
        .map(|b| b.iter().map(|&v| (v as f32) * (v as f32)).sum::<f32>() / (b.len() as f32))
        .collect();
    block_power.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = block_power
        .get(block_power.len() / 10)
        .copied()
        .unwrap_or(0.0)
        .sqrt();

    // Approximately Gaussian noise: sum of four uniforms, unit variance.
    for _ in 0..2 * GAP_PAIRS {
        let mut sum = 0.0f32;
        for _ in 0..4 {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 17;
            *rng ^= *rng << 5;
            sum += (*rng as f32 / u32::MAX as f32) - 0.5;
        }
        out.push((sum * 1.732 * floor).clamp(-32768.0, 32767.0) as i16);
    }
    out
}

/// The ESP_STREAM argument from the environment, with whether it asks for
/// channelized bursts and Wi-Fi rejection.
fn stream_arg() -> Result<(u16, bool, bool), String> {
    let reject = std::env::var_os("BD_ESPDR_KEEP_WIDEBAND").is_none();
    let channelize = std::env::var_os("BD_ESPDR_WIDE").is_none();
    let telemetry = std::env::var_os("BD_ESPDR_TELEMETRY").is_some();
    let trigger_ratio = match std::env::var("BD_ESPDR_TRIGGER_RATIO") {
        Ok(value) => value
            .parse::<u16>()
            .ok()
            .filter(|ratio| (3..=31).contains(ratio))
            .ok_or_else(|| "BD_ESPDR_TRIGGER_RATIO must be an integer from 3 to 31".to_string())?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("BD_ESPDR_TRIGGER_RATIO is not valid text".to_string())
        }
    };
    let arg = if reject { STREAM_REJECT_WIDEBAND } else { 0 }
        | if channelize { STREAM_CHANNELIZE } else { 0 }
        | if telemetry { STREAM_TELEMETRY } else { 0 }
        | trigger_ratio << STREAM_TRIGGER_RATIO_SHIFT;
    if trigger_ratio != 0 {
        eprintln!("eSpDR: burst trigger {}x noise", trigger_ratio);
    }
    Ok((arg, channelize, reject))
}

/// Opens the ESP at `path`, checks its firmware and tunes it, with baseband
/// filter code `filter` at 16 Msps. Returns the link and the LO actually
/// tuned, in Hz.
fn open_board(
    path: &str,
    rate_sel: u16,
    width: u16,
    gain_sel: u16,
    lo_hz: u32,
    filter: u16,
) -> Result<(EspLink, u32), String> {
    let mut link = EspLink::open(path)?;
    let id = link.handshake().map_err(|e| {
        format!(
            "{} (the ESP at {} did not answer: load the firmware into its RAM again, or power-cycle it)",
            e, path
        )
    })?;
    if id != CTL_ESP_FIRMWARE_ID {
        return Err(format!("eSpDR: unexpected firmware id {:#010x} on {}", id, path));
    }
    if link.command(CTL_STATUS, ESP_STAT_RADIO)? != 0 {
        return Err(format!("eSpDR: radio initialisation failed on the ESP at {}", path));
    }
    link.command(ESP_SET_RATE, rate_sel)?;
    link.command(ESP_SET_WIDTH, width)?;
    if rate_sel == 1 {
        link.command(ESP_SET_FILTER, filter | filter << 8)?;
    }
    link.command(ESP_SET_GAIN, gain_sel)?;
    let tuned = link.command32(ESP_SET_LO, lo_hz)?;
    Ok((link, tuned))
}

/// The baseband filter code for 16 Msps: `BD_ESPDR_FILTER` (0..63) or
/// `default`.
fn filter_code(default: u16) -> Result<u16, String> {
    match std::env::var("BD_ESPDR_FILTER") {
        Err(_) => Ok(default),
        Ok(v) => v
            .trim()
            .parse::<u16>()
            .ok()
            .filter(|&c| c <= 63)
            .ok_or_else(|| format!("eSpDR: BD_ESPDR_FILTER must be 0..63, got '{}'", v)),
    }
}

/// Serial ports of the attached ESP32-S3 boards, sorted by path.
fn esp_ports() -> Vec<String> {
    let mut ports: Vec<String> = match std::fs::read_dir("/dev/serial/by-id") {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("usb-Espressif_USB_JTAG_serial_debug_unit"))
            })
            .map(|p| p.to_string_lossy().into_owned())
            .collect(),
        Err(_) => Vec::new(),
    };
    ports.sort();
    ports
}

/// Finds the serial port for an interface string.
fn resolve_port(iface: &str) -> Result<String, String> {
    if let Some(path) = iface.strip_prefix("espdr:") {
        return Ok(path.to_string());
    }
    let index: usize = match iface.strip_prefix("espdr").unwrap_or("") {
        "" => 0,
        n => n
            .parse()
            .map_err(|_| format!("invalid eSpDR interface: '{}'", iface))?,
    };
    let ports = esp_ports();
    ports.get(index).cloned().ok_or_else(|| {
        format!(
            "eSpDR: no ESP32-S3 USB serial port #{} ({} found)",
            index,
            ports.len()
        )
    })
}

#[derive(Default)]
struct ReceiverTelemetry {
    queue_high_water: u32,
    triggers: u64,
    trigger_power: f32,
    trigger_floor: f32,
    rejected: [u64; 3],
    backlog: u32,
}

impl ReceiverTelemetry {
    fn add(&mut self, words: &[u32]) {
        if words.len() < STATUS_V2_WORDS {
            return;
        }
        self.queue_high_water = self.queue_high_water.max(words[8]);
        self.triggers += words[9] as u64;
        let power = f32::from_bits(words[10]);
        if power > self.trigger_power {
            self.trigger_power = power;
            self.trigger_floor = f32::from_bits(words[11]);
        }
        for (total, &value) in self.rejected.iter_mut().zip(&words[12..15]) {
            *total += value as u64;
        }
        self.backlog = self.backlog.max(words[15]);
    }

    fn report(&mut self, board: Option<usize>) {
        let trigger = if self.trigger_floor <= 0.0 {
            0.0
        } else {
            self.trigger_power / self.trigger_floor
        };
        let label = board.map_or_else(|| "eSpDR".to_string(), |n| format!("eSpDR board {}", n));
        eprintln!(
            "{} rx: queue={}/{} triggers={} max={:.2}x reject(mask/env/power)={}/{}/{} backlog={}",
            label,
            self.queue_high_water,
            64 * 1024,
            self.triggers,
            trigger,
            self.rejected[0],
            self.rejected[1],
            self.rejected[2],
            self.backlog,
        );
        *self = Self::default();
    }
}

/// Receives the burst stream until stopped. Gain changes stop the stream,
/// apply the setting and restart it, continuing the timeline.
fn stream_loop(
    mut link: EspLink,
    tx: Sender<Vec<i16>>,
    gain_rx: Receiver<u8>,
    running: &AtomicBool,
    overflow: &AtomicU64,
    arg: u16,
) {
    let mut timeline = Timeline::new(tx);
    let mut telemetry = ReceiverTelemetry::default();
    let mut telemetry_reported = Instant::now();
    while running.load(Ordering::Relaxed) {
        if let Ok(gain) = gain_rx.try_recv() {
            link.stop_stream();
            if let Err(e) = link.command(ESP_SET_GAIN, gain as u16) {
                eprintln!("{}", e);
            }
            match link.start_stream(arg) {
                Ok(true) => timeline.restart(),
                Ok(false) => return,
                Err(e) => {
                    eprintln!("{}", e);
                    return;
                }
            }
        }
        match link.next_record() {
            Ok(Record::Burst { start, pairs }) => {
                if !timeline.burst(start, &convert_pairs(&pairs)) {
                    break; // stop the ESP's stream on the way out
                }
            }
            Ok(Record::Narrow { start, offset, pairs }) => {
                if !timeline.burst(start, &convert_narrow(&pairs, start, offset)) {
                    break; // stop the ESP's stream on the way out
                }
            }
            Ok(Record::Status { start, usb_frame: _, words }) => {
                let floor = f32::from_bits(words[0]);
                if floor > 0.0 {
                    timeline.sigma = (floor / 2.0).sqrt() * SAMPLE_SCALE as f32;
                }
                // Bursts the ESP dropped for queue space, plus blocks skipped
                // when it fell behind.
                overflow.store(words[3] as u64 + words[5] as u64, Ordering::Relaxed);
                telemetry.add(&words);
                if words.len() >= STATUS_V2_WORDS
                    && telemetry_reported.elapsed() >= Duration::from_secs(1)
                {
                    telemetry.report(None);
                    telemetry_reported = Instant::now();
                }
                if !timeline.fill_to(timeline.base + start) {
                    break; // stop the ESP's stream on the way out
                }
            }
            Ok(Record::End) => return,
            Err(e) => {
                // A corrupted record: resynchronise by restarting the stream.
                eprintln!("{}", e);
                link.stop_stream();
                match link.start_stream(arg) {
                    Ok(true) => timeline.restart(),
                    _ => return,
                }
            }
        }
    }
    link.stop_stream();
}

/// Requests snapshots until stopped (firmware without streaming).
fn snapshot_loop(
    mut link: EspLink,
    tx: Sender<Vec<i16>>,
    gain_rx: Receiver<u8>,
    running: &AtomicBool,
    overflow: &AtomicU64,
) {
    let mut rng = 0x1234_5678u32;
    while running.load(Ordering::Relaxed) {
        if let Ok(g) = gain_rx.try_recv() {
            if let Err(e) = link.command(ESP_SET_GAIN, g as u16) {
                eprintln!("{}", e);
            }
        }
        let words = match link.snapshot() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("{}", e);
                link.drain();
                continue;
            }
        };
        match tx.try_send(convert_snapshot(&words, &mut rng)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                overflow.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => break,
        }
    }
}

pub struct EspdrHandle {
    rx: Receiver<Vec<i16>>,
    pending: Vec<i16>,
    pending_offset: usize,
    max_samps: usize,
    running: Arc<AtomicBool>,
    overflow: Arc<AtomicU64>,
    gain_tx: Sender<u8>,
    thread: Option<JoinHandle<()>>,
    lo_hz: u32,
}

impl EspdrHandle {
    /// Opens the ESP, configures the receiver and starts receiving, streaming
    /// bursts if the firmware supports it and falling back to snapshots.
    /// `sample_rate` must be 16 or 80 Msps; `gain` is the ESP gain-table
    /// selector (0..127, not dB; about 24..30 suits a nearby antenna).
    pub fn open(iface: &str, sample_rate: u32, center_freq: u64, gain: i32) -> Result<Self, String> {
        if let Some(paths) = multi::boards_for(iface, sample_rate)? {
            return multi::open(&paths, center_freq, gain);
        }
        let (rate_sel, width) = match sample_rate {
            16_000_000 => (1u16, 20u16),
            80_000_000 => (0u16, 40u16),
            _ => {
                return Err(format!(
                    "eSpDR sample rate must be 16 or 80 Msps (-C 16 or -C 80), got {}",
                    sample_rate
                ))
            }
        };
        let path = resolve_port(iface)?;
        let gain_sel = gain.clamp(0, 127) as u16;
        let (mut link, lo_hz) =
            open_board(&path, rate_sel, width, gain_sel, center_freq as u32, filter_code(FILTER_16M)?)?;
        let (stream_arg, channelize, reject) = stream_arg()?;
        let force_snapshot = std::env::var_os("BD_ESPDR_SNAPSHOT").is_some();
        let streaming = !force_snapshot && link.start_stream(stream_arg)?;
        if !streaming {
            // Older firmware, or 80 Msps: verify snapshots before relying on them.
            link.snapshot()?;
        }
        eprintln!(
            "eSpDR: {} LO {:.6} MHz, {} Msps, gain selector {}, {}",
            path,
            lo_hz as f64 / 1e6,
            sample_rate / 1_000_000,
            gain_sel,
            if !streaming {
                "snapshot mode"
            } else {
                match (channelize, reject) {
                    (true, true) => "streaming channelized bursts (Wi-Fi rejected)",
                    (true, false) => "streaming channelized bursts",
                    (false, true) => "streaming bursts (Wi-Fi rejected)",
                    (false, false) => "streaming bursts",
                }
            }
        );

        let (tx, rx) = bounded::<Vec<i16>>(64);
        let (gain_tx, gain_rx) = bounded::<u8>(4);
        let running = Arc::new(AtomicBool::new(true));
        let overflow = Arc::new(AtomicU64::new(0));
        let thread = {
            let running = running.clone();
            let overflow = overflow.clone();
            std::thread::Builder::new()
                .name("espdr-recv".to_string())
                .spawn(move || {
                    if streaming {
                        stream_loop(link, tx, gain_rx, &running, &overflow, stream_arg);
                    } else {
                        snapshot_loop(link, tx, gain_rx, &running, &overflow);
                    }
                })
                .map_err(|e| format!("eSpDR: cannot start receive thread: {}", e))?
        };

        Ok(Self {
            rx,
            pending: Vec::new(),
            pending_offset: 0,
            max_samps: CHUNK_PAIRS,
            running,
            overflow,
            gain_tx,
            thread: Some(thread),
            lo_hz,
        })
    }

    /// Fills `buf` with interleaved I/Q and returns the complex samples written.
    pub fn recv_into_i16(&mut self, buf: &mut [i16]) -> usize {
        let max = buf.len() & !1;
        let mut written = 0;
        while written < max {
            if self.pending_offset >= self.pending.len() {
                match self.rx.recv_timeout(Duration::from_secs(3)) {
                    Ok(chunk) => {
                        self.pending = chunk;
                        self.pending_offset = 0;
                    }
                    Err(_) => break,
                }
            }
            let n = (self.pending.len() - self.pending_offset).min(max - written);
            buf[written..written + n]
                .copy_from_slice(&self.pending[self.pending_offset..self.pending_offset + n]);
            written += n;
            self.pending_offset += n;
            if self.pending_offset >= self.pending.len() {
                break; // return one chunk at a time
            }
        }
        written / 2
    }

    /// Changes the ESP gain-table selector (0..127).
    pub fn set_gain(&self, gain: i32) {
        let _ = self.gain_tx.try_send(gain.clamp(0, 127) as u8);
    }

    pub fn max_samps(&self) -> usize {
        self.max_samps
    }

    /// Data lost before reaching the pipeline: in stream mode, bursts the ESP
    /// dropped plus blocks it skipped; in snapshot mode, snapshots dropped
    /// because the pipeline fell behind.
    pub fn overflow_count(&self) -> u64 {
        self.overflow.load(Ordering::Relaxed)
    }

    /// The LO frequency the ESP actually tuned, in Hz.
    pub fn lo_hz(&self) -> u32 {
        self.lo_hz
    }
}

impl Drop for EspdrHandle {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        // Close the channel first, so a thread blocked handing over a chunk
        // sees the pipeline has gone instead of waiting for it.
        let (_, closed) = bounded(1);
        drop(std::mem::replace(&mut self.rx, closed));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn conversion_conjugates_and_scales() {
        // I = +100, Q = -50 as 10-bit two's complement.
        let i = 100u32;
        let q = (-50i32 as u32) & 1023;
        let word = i | (q << 10);
        let mut rng = 1;
        let out = convert_snapshot(&[word], &mut rng);
        assert_eq!(out[0], 100 * 64);
        assert_eq!(out[1], 50 * 64); // Q negated by the conjugation
        assert_eq!(out.len(), 2 + 2 * GAP_PAIRS);
    }

    #[test]
    fn interface_with_explicit_path() {
        assert_eq!(resolve_port("espdr:/dev/ttyACM3").unwrap(), "/dev/ttyACM3");
        assert!(resolve_port("espdrX").is_err());
    }

    #[test]
    fn extended_status_accumulates() {
        let mut status = vec![0; STATUS_V2_WORDS];
        status[8] = 12000;
        status[9] = 7;
        status[10] = 20.0f32.to_bits();
        status[11] = 4.0f32.to_bits();
        status[12..16].copy_from_slice(&[1, 2, 3, 5000]);
        let mut telemetry = ReceiverTelemetry::default();
        telemetry.add(&status);
        assert_eq!(telemetry.queue_high_water, 12000);
        assert_eq!(telemetry.triggers, 7);
        assert_eq!(telemetry.rejected, [1, 2, 3]);
        assert_eq!(telemetry.backlog, 5000);
        assert_eq!(telemetry.trigger_power / telemetry.trigger_floor, 5.0);
    }

    fn drain(rx: &Receiver<Vec<i16>>) -> Vec<i16> {
        let mut all = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            all.extend(chunk);
        }
        all
    }

    #[test]
    fn timeline_places_bursts_on_sample_time() {
        let (tx, rx) = bounded::<Vec<i16>>(1024);
        let mut t = Timeline::new(tx);
        let burst = [100u32; 10]; // I = 100, Q = 0
        assert!(t.burst(100, &convert_pairs(&burst)));
        let out = drain(&rx);
        assert_eq!(out.len(), 2 * 110);
        assert_eq!(out[2 * 100], 100 * 64); // the burst starts exactly at pair 100
        assert_eq!(t.emitted, 110);
    }

    #[test]
    fn timeline_trims_overlap_and_continues_after_restart() {
        let (tx, rx) = bounded::<Vec<i16>>(1024);
        let mut t = Timeline::new(tx);
        assert!(t.burst(0, &convert_pairs(&[1u32; 20])));
        assert!(t.burst(15, &convert_pairs(&[2u32; 10]))); // overlaps the first by 5 pairs
        assert_eq!(t.emitted, 25);
        t.restart(); // a new stream counts from 0 again
        assert!(t.burst(5, &convert_pairs(&[3u32; 1])));
        assert_eq!(t.emitted, 31);
        let out = drain(&rx);
        assert_eq!(out.len(), 2 * 31);
        assert_eq!(out[2 * 30], 3 * 64);
    }

    #[test]
    fn timeline_caps_long_gaps() {
        let (tx, rx) = bounded::<Vec<i16>>(4096);
        let mut t = Timeline::new(tx);
        assert!(t.fill_to(MAX_GAP_PAIRS * 3));
        assert_eq!(t.emitted, MAX_GAP_PAIRS * 3);
        assert_eq!(drain(&rx).len() as u64, 2 * MAX_GAP_PAIRS);
    }

    #[test]
    fn remove_dc_takes_out_the_offset_and_keeps_the_channel() {
        // A record mixed down by 1 MHz: the board's DC is a tone at -1 MHz
        // (a quarter turn back per 4 Msps sample), the channel a tone at
        // -0.5 MHz, half a MHz from the DC as a tiled Bluetooth channel is.
        let n = 400;
        let tone = |f_mhz: f64, amp: f64, k: usize| {
            let a = 2.0 * std::f64::consts::PI * f_mhz * k as f64 / 4.0;
            (amp * a.cos(), amp * a.sin())
        };
        let power_at = |y: &[(f32, f32)], f: f64| {
            let (mut i, mut q) = (0.0, 0.0);
            for (k, &(a, b)) in y.iter().enumerate() {
                let (c, s) = tone(-f, 1.0, k);
                i += a as f64 * c - b as f64 * s;
                q += a as f64 * s + b as f64 * c;
            }
            (i * i + q * q) / (y.len() * y.len()) as f64
        };
        // Mixed down by +1 or -1 MHz, the DC lies at -1 or +1 MHz.
        for offset in [1, -1] {
            let dc = -offset as f64;
            let channel = dc / 2.0;
            let mut y: Vec<(f32, f32)> = (0..n)
                .map(|k| {
                    let (di, dq) = tone(dc, 7.0, k);
                    let (si, sq) = tone(channel, 3.0, k);
                    ((di + si) as f32, (dq + sq) as f32)
                })
                .collect();
            remove_dc(&mut y, offset);
            assert!(power_at(&y, dc) < 1e-6, "offset {}: DC left: {}", offset, power_at(&y, dc));
            assert!((power_at(&y, channel) - 9.0).abs() < 0.1, "offset {}: channel changed", offset);
        }
        // Far from the window the record is left alone.
        let mut far = vec![(5.0f32, -2.0f32); n];
        remove_dc(&mut far, 3);
        assert_eq!(far[0], (5.0, -2.0));
    }

    #[test]
    fn narrow_bursts_return_to_their_channel() {
        // A steady signal in a channel 3 MHz from the LO (LO-minus-RF), as
        // the ESP would send it after mixing down: constant I = 100.
        let pairs = [100u32; 64];
        let start = 1000u64;
        let out = convert_narrow(&pairs, start, 3);
        assert_eq!(out.len(), 2 * 4 * 64);
        // Away from the edges the restored pairs rotate by 3/16 of a turn
        // per pair, and conjugation turns the rotation the other way.
        for i in 20..200 {
            let a = 2.0 * std::f32::consts::PI * (3 * (start + i as u64) % 16) as f32 / 16.0;
            let (ei, eq) = (6400.0 * a.cos(), -6400.0 * a.sin());
            assert!((out[2 * i] as f32 - ei).abs() <= 64.0 * 2.0, "I at {}: {} vs {}", i, out[2 * i], ei);
            assert!((out[2 * i + 1] as f32 - eq).abs() <= 64.0 * 2.0, "Q at {}: {} vs {}", i, out[2 * i + 1], eq);
        }
    }
}
