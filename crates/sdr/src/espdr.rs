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
//!   at the reported floor. Bursts whose envelope fluctuates like OFDM
//!   (Wi-Fi) are dropped on the ESP unless `BD_ESPDR_KEEP_WIDEBAND` is set.
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

use crossbeam::channel::{bounded, Receiver, Sender, TrySendError};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

const REQ_MAGIC: u8 = 0xB4;
const RSP_MAGIC: u8 = 0xB5;
const CTL_INFO: u8 = 1;
const CTL_STATUS: u8 = 3;
const ESP_ARG_HIGH: u8 = 19;
const ESP_SET_LO: u8 = 20;
const ESP_SET_RATE: u8 = 21;
const ESP_SET_WIDTH: u8 = 22;
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
const STREAM_REJECT_WIDEBAND: u16 = 1;
const STATUS_WORDS: usize = 8;
/// Largest stretch of noise inserted for one gap (0.1 s at 16 Msps); a longer
/// gap only happens across a restart, and its time is skipped.
const MAX_GAP_PAIRS: u64 = 1_600_000;
/// Pairs per chunk handed to the pipeline.
const CHUNK_PAIRS: usize = 32768;

/// Noise-filled pairs placed between snapshots.
const GAP_PAIRS: usize = 2048;
/// Scale from 10-bit samples toward int16 full scale.
const SAMPLE_SCALE: i32 = 64;

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
    Status { start: u64, words: [u32; STATUS_WORDS] },
    End,
}

impl EspLink {
    /// Starts burst streaming. Ok(false) if the firmware cannot stream (no
    /// command, or not at this sample rate).
    fn start_stream(&mut self, arg: u16) -> Result<bool, String> {
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
        let start = word(8) as u64 | (word(12) as u64) << 32;
        let length = word(16) as usize;
        let mut check = [0u8; 4];
        match kind {
            STREAM_BURST => {
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
                Ok(Record::Burst { start, pairs })
            }
            STREAM_STATUS => {
                let mut payload = [0u8; STATUS_WORDS * 4];
                self.read_exact(&mut payload)?;
                self.read_exact(&mut check)?;
                let mut words = [0u32; STATUS_WORDS];
                for (w, b) in words.iter_mut().zip(payload.chunks_exact(4)) {
                    *w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                let sum = words.iter().fold(0u32, |a, &w| a.wrapping_add(w));
                if sum != u32::from_le_bytes(check) {
                    return Err("eSpDR: stream status check failed".to_string());
                }
                Ok(Record::Status { start, words })
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

    /// Adds a burst that starts at stream pair `start`.
    fn burst(&mut self, start: u64, pairs: &[u32]) -> bool {
        let at = self.base + start;
        if !self.fill_to(at) {
            return false;
        }
        let skip = (self.emitted.saturating_sub(at) as usize).min(pairs.len());
        for piece in pairs[skip..].chunks(CHUNK_PAIRS) {
            if !self.send(convert_pairs(piece)) {
                return false;
            }
        }
        self.emitted = self.emitted.max(at + pairs.len() as u64);
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
    let mut ports: Vec<String> = std::fs::read_dir("/dev/serial/by-id")
        .map_err(|_| "eSpDR: no USB serial devices found".to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("usb-Espressif_USB_JTAG_serial_debug_unit"))
        })
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    ports.sort();
    ports.get(index).cloned().ok_or_else(|| {
        format!(
            "eSpDR: no ESP32-S3 USB serial port #{} ({} found)",
            index,
            ports.len()
        )
    })
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
                if !timeline.burst(start, &pairs) {
                    return;
                }
            }
            Ok(Record::Status { start, words }) => {
                let floor = f32::from_bits(words[0]);
                if floor > 0.0 {
                    timeline.sigma = (floor / 2.0).sqrt() * SAMPLE_SCALE as f32;
                }
                // Bursts the ESP dropped for queue space, plus blocks skipped
                // when it fell behind.
                overflow.store(words[3] as u64 + words[5] as u64, Ordering::Relaxed);
                if !timeline.fill_to(timeline.base + start) {
                    return;
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
        let mut link = EspLink::open(&path)?;

        let id = link.handshake()?;
        if id != CTL_ESP_FIRMWARE_ID {
            return Err(format!("eSpDR: unexpected firmware id {:#010x} on {}", id, path));
        }
        if link.command(CTL_STATUS, ESP_STAT_RADIO)? != 0 {
            return Err("eSpDR: radio initialisation failed on the ESP".to_string());
        }
        link.command(ESP_SET_RATE, rate_sel)?;
        link.command(ESP_SET_WIDTH, width)?;
        let gain_sel = gain.clamp(0, 127) as u16;
        link.command(ESP_SET_GAIN, gain_sel)?;
        let lo_hz = link.command32(ESP_SET_LO, center_freq as u32)?;
        let reject = std::env::var_os("BD_ESPDR_KEEP_WIDEBAND").is_none();
        let stream_arg = if reject { STREAM_REJECT_WIDEBAND } else { 0 };
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
            } else if reject {
                "streaming bursts (Wi-Fi rejected)"
            } else {
                "streaming bursts"
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
        assert!(t.burst(100, &burst));
        let out = drain(&rx);
        assert_eq!(out.len(), 2 * 110);
        assert_eq!(out[2 * 100], 100 * 64); // the burst starts exactly at pair 100
        assert_eq!(t.emitted, 110);
    }

    #[test]
    fn timeline_trims_overlap_and_continues_after_restart() {
        let (tx, rx) = bounded::<Vec<i16>>(1024);
        let mut t = Timeline::new(tx);
        assert!(t.burst(0, &[1u32; 20]));
        assert!(t.burst(15, &[2u32; 10])); // overlaps the first by 5 pairs
        assert_eq!(t.emitted, 25);
        t.restart(); // a new stream counts from 0 again
        assert!(t.burst(5, &[3u32; 1]));
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
}
