// Copyright 2025-2026 CEMAXECUTER LLC

//! One or more ESP32-S3 receivers as an 80 MHz receiver.
//!
//! At 16 Msps an ESP's samples are its 80 Msps capture decimated by five
//! without filtering, so it hears about 80 MHz around its LO folded into a
//! 16 MHz window: a burst seen at some offset in the window may have come
//! from that offset or from 16, 32 MHz either side of it, at much the same
//! strength. There is nothing in the samples to tell which, so each burst is
//! placed at every frequency it could have come from on an 80 Msps timeline,
//! and the decoders sort them out: a BLE packet only passes its CRC when it
//! is dewhitened for the channel it was sent on.
//!
//! All the ESPs tune to the same LO and hear the same folded band. Each one
//! sends only bursts in its share of the channel positions in the window
//! (the firmware's channel mask), so together they carry several times what
//! one USB link can, and bursts at different positions that overlap in time
//! are no longer lost to a single receiver's one-burst-at-a-time detector.
//!
//! Each board counts samples on its own crystal, so board time is mapped to
//! host time: a record cannot arrive before the samples in it were taken, so
//! the earliest arrival seen relative to its last sample tracks when the
//! board's count began (within about a millisecond of USB scheduling), and
//! that estimate is allowed to creep later at 50 ppm to follow a slower
//! crystal. The timeline runs `DELAY` behind real time so bursts still queued
//! on an ESP arrive before their place is emitted.

use super::*;
use num_complex::Complex32;
use std::time::Instant;

/// Board sample rate.
const BOARD_RATE: f64 = 16e6;
/// Output sample rate, the span an ESP hears around its LO.
const FOLD_RATE: u32 = 80_000_000;
/// Output samples per board pair.
const FACTOR: usize = 5;
/// How far from the LO (MHz) a folded signal is still heard: measured flat
/// to about 25 MHz, 5 dB down at 39 and gone by 55.
const HEARING_MHZ: i64 = 40;
/// How far the output timeline runs behind real time.
const DELAY: f64 = 0.4;
/// How fast a board's start estimate may move later, in seconds per second.
const DRIFT_ALLOWANCE: f64 = 50e-6;
/// Interpolation taps per output phase, for a 4 Msps channel and for a
/// 16 Msps window.
const NARROW_TAPS_PER_PHASE: usize = 14;
const WIDE_TAPS_PER_PHASE: usize = 48;
/// Output pairs per chunk handed to the pipeline.
const MULTI_CHUNK_PAIRS: usize = 1 << 18;
/// Complex samples in the background noise table.
const NOISE_PAIRS: usize = 1 << 18;

/// The serial ports to use for `iface` at `sample_rate`, or None for a
/// single-board interface. `espdr` takes every attached ESP; a comma list
/// names them.
pub(super) fn boards_for(iface: &str, sample_rate: u32) -> Result<Option<Vec<String>>, String> {
    let list = iface.contains(',');
    if !list && (iface != "espdr" || sample_rate <= 16_000_000) {
        return Ok(None);
    }
    if sample_rate != FOLD_RATE {
        return Err(format!(
            "eSpDR: each ESP hears 80 MHz folded into 16; use -C 80 with -i {} (got -C {})",
            iface,
            sample_rate / 1_000_000
        ));
    }
    let paths: Vec<String> = if list {
        iface
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(resolve_port)
            .collect::<Result<_, _>>()?
    } else {
        esp_ports()
    };
    if paths.is_empty() {
        return Err("eSpDR: no ESP32-S3 USB serial ports found".to_string());
    }
    Ok(Some(paths))
}

/// Per-board gain selectors: `BD_ESPDR_GAINS` (comma list, one per board)
/// or `gain` for all.
fn board_gains(count: usize, gain: i32) -> Result<Vec<u16>, String> {
    let all = gain.clamp(0, 127) as u16;
    let Ok(list) = std::env::var("BD_ESPDR_GAINS") else {
        return Ok(vec![all; count]);
    };
    let gains: Vec<u16> = list
        .split(',')
        .map(|g| g.trim().parse::<u16>().map(|g| g.min(127)))
        .collect::<Result<_, _>>()
        .map_err(|_| format!("eSpDR: BD_ESPDR_GAINS must be numbers, got '{}'", list))?;
    if gains.len() != count {
        return Err(format!("eSpDR: BD_ESPDR_GAINS has {} gains for {} boards", gains.len(), count));
    }
    Ok(gains)
}

/// Channel positions (firmware mask, bit k + 8 for offset k) for board
/// `index` of `count`: positions dealt round, so neighbouring positions go
/// to different boards.
fn position_mask(index: usize, count: usize) -> u16 {
    if count == 1 {
        return 0;
    }
    (0..16).filter(|bit| bit % count == index).fold(0, |m, bit| m | 1 << bit)
}

pub(super) fn open(paths: &[String], center_freq: u64, gain: i32) -> Result<EspdrHandle, String> {
    let count = paths.len();
    let gains = board_gains(count, gain)?;
    let (stream_arg, channelize, reject) = stream_arg();
    let t0 = Instant::now();
    let (event_tx, event_rx) = bounded::<Event>(4096);
    let running = Arc::new(AtomicBool::new(true));
    let mut board_gain_tx = Vec::with_capacity(count);
    let mut threads = Vec::with_capacity(count);
    let mut lo_hz = center_freq as u32;
    for (index, path) in paths.iter().enumerate() {
        let (mut link, tuned) = open_board(path, 1, 20, gains[index], center_freq as u32)?;
        lo_hz = tuned;
        // Whole-window bursts cannot be shared by position.
        let mask = if channelize { position_mask(index, count) } else { 0 };
        if !link.start_stream_masked(stream_arg, mask)? {
            return Err(format!("eSpDR: the ESP at {} cannot stream; load the current firmware", path));
        }
        eprintln!(
            "eSpDR: board {} {} LO {:.6} MHz, gain selector {}, channel positions {:#06x}",
            index,
            path,
            tuned as f64 / 1e6,
            gains[index],
            if mask == 0 { 0xFFFF } else { mask }
        );
        let (gain_tx, gain_rx) = bounded::<u8>(4);
        board_gain_tx.push(gain_tx);
        let events = event_tx.clone();
        let running = running.clone();
        let shaper = Shaper::new();
        threads.push(
            std::thread::Builder::new()
                .name(format!("espdr-board{}", index))
                .spawn(move || board_loop(index, link, shaper, events, gain_rx, &running, stream_arg, mask))
                .map_err(|e| format!("eSpDR: cannot start receive thread: {}", e))?,
        );
    }
    drop(event_tx);
    eprintln!(
        "eSpDR: {} board{} hearing 80 MHz around {:.3} MHz folded into 16, {}",
        count,
        if count == 1 { "" } else { "s" },
        lo_hz as f64 / 1e6,
        match (channelize, reject) {
            (true, true) => "channelized bursts (Wi-Fi rejected)",
            (true, false) => "channelized bursts",
            (false, true) => "bursts (Wi-Fi rejected)",
            (false, false) => "bursts",
        }
    );
    if !channelize && count > 1 {
        eprintln!("eSpDR: whole-window bursts are not shared out, so every board sends the same ones");
    }

    let (tx, rx) = bounded::<Vec<i16>>(64);
    let (gain_tx, gain_rx) = bounded::<u8>(4);
    let overflow = Arc::new(AtomicU64::new(0));
    let merger = {
        let running = running.clone();
        let overflow = overflow.clone();
        std::thread::Builder::new()
            .name("espdr-merge".to_string())
            .spawn(move || {
                let mut merger = Merger::new(count, t0, tx);
                merger.run(&event_rx, &gain_rx, &board_gain_tx, &running, &overflow);
                running.store(false, Ordering::Relaxed);
                drop(event_rx); // unblocks boards waiting to hand over a burst
                for thread in threads {
                    let _ = thread.join();
                }
            })
            .map_err(|e| format!("eSpDR: cannot start merge thread: {}", e))?
    };

    Ok(EspdrHandle {
        rx,
        pending: Vec::new(),
        pending_offset: 0,
        max_samps: MULTI_CHUNK_PAIRS,
        running,
        overflow,
        gain_tx,
        thread: Some(merger),
        lo_hz,
    })
}

enum Event {
    /// A burst ready for the timeline, from a record received at `seen`
    /// whose last pair was board pair `last`.
    Burst { board: usize, seen: Instant, last: u64, burst: Shaped },
    Status { board: usize, seen: Instant, at: u64, words: [u32; STATUS_WORDS] },
    /// The board's stream restarted and counts from 0 again.
    Restart { board: usize },
}

/// Receives one board's stream, shapes its bursts and forwards them until
/// stopped.
#[allow(clippy::too_many_arguments)]
fn board_loop(
    board: usize,
    mut link: EspLink,
    shaper: Shaper,
    events: Sender<Event>,
    gain_rx: Receiver<u8>,
    running: &AtomicBool,
    arg: u16,
    mask: u16,
) {
    let send = |event: Event| -> bool {
        let mut event = event;
        loop {
            match events.send_timeout(event, Duration::from_millis(200)) {
                Ok(()) => return true,
                Err(crossbeam::channel::SendTimeoutError::Timeout(e)) if running.load(Ordering::Relaxed) => event = e,
                Err(_) => return false,
            }
        }
    };
    while running.load(Ordering::Relaxed) {
        if let Ok(gain) = gain_rx.try_recv() {
            link.stop_stream();
            if let Err(e) = link.command(ESP_SET_GAIN, gain as u16) {
                eprintln!("eSpDR board {}: {}", board, e);
            }
            if !matches!(link.start_stream_masked(arg, mask), Ok(true)) || !send(Event::Restart { board }) {
                return;
            }
        }
        let event = match link.next_record() {
            Ok(Record::End) => return,
            Ok(Record::Status { start, words }) => Event::Status { board, seen: Instant::now(), at: start, words },
            Ok(Record::Narrow { start, offset, pairs }) => Event::Burst {
                board,
                seen: Instant::now(),
                last: start + 4 * pairs.len() as u64,
                burst: shaper.narrow(start, offset, &pairs),
            },
            Ok(Record::Burst { start, pairs }) => Event::Burst {
                board,
                seen: Instant::now(),
                last: start + pairs.len() as u64,
                burst: shaper.wide(start, &pairs),
            },
            Err(e) => {
                // A corrupted record: resynchronise by restarting the stream.
                eprintln!("eSpDR board {}: {}", board, e);
                link.stop_stream();
                if !matches!(link.start_stream_masked(arg, mask), Ok(true)) {
                    return;
                }
                Event::Restart { board }
            }
        };
        if !send(event) {
            break;
        }
    }
    link.stop_stream();
}

/// A burst at the output rate, already at every frequency it could have
/// come from.
struct Shaped {
    /// Board pair (fractional) that output sample 0 corresponds to.
    first: f64,
    samples: Vec<Complex32>,
}

/// Interpolates bursts to the output rate and repeats them at each
/// frequency they could have come from.
struct Shaper {
    /// e^(j 2 pi m / 80): whole-MHz offsets at 80 Msps repeat every 80
    /// samples.
    turn: Vec<Complex32>,
    /// From a 4 Msps channel: 20 phases of NARROW_TAPS_PER_PHASE.
    narrow_taps: Vec<Vec<f32>>,
    /// From the 16 Msps window: 5 phases of WIDE_TAPS_PER_PHASE.
    wide_taps: Vec<Vec<f32>>,
}

impl Shaper {
    fn new() -> Self {
        let period = (FOLD_RATE / 1_000_000) as usize;
        let turn = (0..period)
            .map(|m| {
                let a = 2.0 * std::f64::consts::PI * m as f64 / period as f64;
                Complex32::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let rate = period as f64; // output rate in MHz
        // A channel's content lies within +-1.6 MHz and its first image
        // starts 2.4 MHz out; a window's lies within +-8 MHz, imaged from
        // 8 MHz on, so keep it to about +-7.4.
        let narrow_taps = polyphase(lowpass(4 * FACTOR, NARROW_TAPS_PER_PHASE, 2.0 / rate, 4.5), 4 * FACTOR);
        let wide_taps = polyphase(lowpass(FACTOR, WIDE_TAPS_PER_PHASE, 8.0 / rate, 5.65), FACTOR);
        Self { turn, narrow_taps, wide_taps }
    }

    /// The offsets from the LO (MHz) that a signal seen at `offset` in the
    /// window could have come from: it, and every 16 MHz either side that
    /// the ESP hears and the output holds.
    fn folds(offset: i64) -> Vec<i64> {
        let half = (FOLD_RATE / 2_000_000) as i64;
        (-4..=4)
            .map(|m| offset + 16 * m)
            .filter(|f| f.abs() <= HEARING_MHZ && (-half..half).contains(f))
            .collect()
    }

    /// A channelized record: outputs every fourth board pair from `start`
    /// (centred on start + 4j - 2.5), mixed down by `offset` MHz in the
    /// LO-minus-RF orientation, i.e. the channel lies at LO - offset.
    fn narrow(&self, start: u64, offset: i32, pairs: &[u32]) -> Shaped {
        let x: Vec<Complex32> = pairs
            .iter()
            .map(|&w| {
                let i = ((w & 1023) ^ 512) as f32 - 512.0;
                let q = (((w >> 10) & 1023) ^ 512) as f32 - 512.0;
                // Conjugate to RF-minus-LO and scale like convert_pairs.
                Complex32::new(i, -q) * SAMPLE_SCALE as f32
            })
            .collect();
        let step = 1.0 / FACTOR as f64; // board pairs per output sample
        let delay = (self.narrow_taps.len() * NARROW_TAPS_PER_PHASE - 1) as f64 / 2.0;
        Shaped {
            first: start as f64 - 2.5 - delay * step,
            samples: self.interpolate(&x, &self.narrow_taps, &Self::folds(-offset as i64)),
        }
    }

    /// A whole-window record starting at board pair `start`.
    fn wide(&self, start: u64, pairs: &[u32]) -> Shaped {
        let x: Vec<Complex32> = convert_pairs(pairs)
            .chunks_exact(2)
            .map(|p| Complex32::new(p[0] as f32, p[1] as f32))
            .collect();
        let delay = (self.wide_taps.len() * WIDE_TAPS_PER_PHASE - 1) as f64 / 2.0;
        Shaped {
            first: start as f64 - delay / FACTOR as f64,
            samples: self.interpolate(&x, &self.wide_taps, &Self::folds(0)),
        }
    }

    /// Interpolates `x` by the number of phases and repeats it at each of
    /// `offsets` (MHz).
    fn interpolate(&self, x: &[Complex32], phases: &[Vec<f32>], offsets: &[i64]) -> Vec<Complex32> {
        let ratio = phases.len();
        let taps = phases[0].len();
        let period = self.turn.len();
        // The sum of the offsets' mixers repeats every `period` samples.
        let mixer: Vec<Complex32> = (0..period)
            .map(|g| {
                offsets
                    .iter()
                    .map(|&f| self.turn[(f * g as i64).rem_euclid(period as i64) as usize])
                    .sum()
            })
            .collect();
        let mut padded = vec![Complex32::new(0.0, 0.0); x.len() + 2 * taps];
        padded[taps..taps + x.len()].copy_from_slice(x);
        let mut g = 0usize;
        let mut out = Vec::with_capacity((x.len() + taps) * ratio);
        for i in 0..x.len() + taps {
            // Inputs i, i-1, ... i-taps+1, newest first.
            let window = &padded[i + 1..i + taps + 1];
            for h in phases {
                let mut acc = Complex32::new(0.0, 0.0);
                for (v, &t) in window.iter().rev().zip(h) {
                    acc += v * t;
                }
                out.push(acc * mixer[g]);
                g += 1;
                if g == period {
                    g = 0;
                }
            }
        }
        out
    }
}

/// Kaiser-windowed low-pass of `ratio * per_phase` taps for interpolating
/// by `ratio`, cutting off at `cutoff` cycles per output sample, with gain
/// `ratio` to keep the level.
fn lowpass(ratio: usize, per_phase: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let len = ratio * per_phase;
    let bessel_i0 = |x: f64| {
        let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
        while term > 1e-12 * sum {
            term *= (x / (2.0 * k)).powi(2);
            sum += term;
            k += 1.0;
        }
        sum
    };
    let mid = (len - 1) as f64 / 2.0;
    let mut taps: Vec<f64> = (0..len)
        .map(|i| {
            let t = i as f64 - mid;
            let sinc = if t == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * std::f64::consts::PI * cutoff * t).sin() / (std::f64::consts::PI * t)
            };
            sinc * bessel_i0(beta * (1.0 - (t / (mid + 1.0)).powi(2)).sqrt()) / bessel_i0(beta)
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    for t in &mut taps {
        *t *= ratio as f64 / sum;
    }
    taps
}

/// Splits interpolation taps into `ratio` phases: output r of each input
/// step uses taps r, r + ratio, ... against the newest input first.
fn polyphase(taps: Vec<f64>, ratio: usize) -> Vec<Vec<f32>> {
    (0..ratio)
        .map(|r| taps.iter().skip(r).step_by(ratio).map(|&t| t as f32).collect())
        .collect()
}

struct Board {
    /// Host time (seconds since start) at which the board's pair 0 was taken.
    start: Option<f64>,
    last_seen: f64,
    /// Background noise per component, in output units.
    sigma: f32,
    /// Bursts dropped and blocks skipped on the ESP, from its last status.
    lost: u64,
}

/// A burst placed on the output timeline.
struct Placed {
    at: i64,
    samples: Vec<Complex32>,
}

struct Merger {
    boards: Vec<Board>,
    rate: f64,
    t0: Instant,
    tx: Sender<Vec<i16>>,
    placed: Vec<Placed>,
    emitted: i64,
    /// Unit-variance Gaussian noise, and the same scaled to `noise_sigma`.
    unit_noise: Vec<f32>,
    noise: Vec<i16>,
    noise_sigma: f32,
    noise_pos: usize,
    late: u64,
}

impl Merger {
    fn new(count: usize, t0: Instant, tx: Sender<Vec<i16>>) -> Self {
        let boards = (0..count)
            .map(|_| Board { start: None, last_seen: 0.0, sigma: 4.0 * SAMPLE_SCALE as f32, lost: 0 })
            .collect();
        let mut state = 0x2545_F491u32;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 + 1.0) / (u32::MAX as f32 + 2.0)
        };
        let mut unit_noise = Vec::with_capacity(2 * NOISE_PAIRS);
        while unit_noise.len() < 2 * NOISE_PAIRS {
            let r = (-2.0 * uniform().ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * uniform();
            unit_noise.push(r * theta.cos());
            unit_noise.push(r * theta.sin());
        }
        Self {
            boards,
            rate: FOLD_RATE as f64,
            t0,
            tx,
            placed: Vec::new(),
            emitted: 0,
            unit_noise,
            noise: Vec::new(),
            noise_sigma: -1.0,
            noise_pos: 0,
            late: 0,
        }
    }

    fn run(
        &mut self,
        events: &Receiver<Event>,
        gain_rx: &Receiver<u8>,
        board_gain_tx: &[Sender<u8>],
        running: &AtomicBool,
        overflow: &AtomicU64,
    ) {
        while running.load(Ordering::Relaxed) {
            if let Ok(gain) = gain_rx.try_recv() {
                for tx in board_gain_tx {
                    let _ = tx.try_send(gain);
                }
            }
            match events.recv_timeout(Duration::from_millis(2)) {
                Ok(event) => self.take(event),
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {}
                Err(_) => return,
            }
            while let Ok(event) = events.try_recv() {
                self.take(event);
            }
            let lost: u64 = self.boards.iter().map(|b| b.lost).sum();
            overflow.store(lost + self.late, Ordering::Relaxed);
            let now = self.t0.elapsed().as_secs_f64();
            let target = ((now - DELAY) * self.rate) as i64;
            if target - self.emitted >= MULTI_CHUNK_PAIRS as i64 && !self.emit(MULTI_CHUNK_PAIRS, running) {
                return;
            }
        }
    }

    /// Updates the board's start estimate from a record's arrival.
    fn track(&mut self, board: usize, seen: Instant, last: u64) {
        let seen = seen.duration_since(self.t0).as_secs_f64();
        let b = &mut self.boards[board];
        let candidate = seen - last as f64 / BOARD_RATE;
        b.start = Some(match b.start {
            None => candidate,
            Some(s) => (s + DRIFT_ALLOWANCE * (seen - b.last_seen)).min(candidate),
        });
        b.last_seen = seen;
    }

    fn take(&mut self, event: Event) {
        match event {
            Event::Restart { board } => self.boards[board].start = None,
            Event::Status { board, seen, at, words } => {
                self.track(board, seen, at);
                let b = &mut self.boards[board];
                let floor = f32::from_bits(words[0]);
                if floor > 0.0 {
                    b.sigma = (floor / 2.0).sqrt() * SAMPLE_SCALE as f32;
                }
                b.lost = words[3] as u64 + words[5] as u64;
            }
            Event::Burst { board, seen, last, burst } => {
                self.track(board, seen, last);
                let start = self.boards[board].start.unwrap_or(0.0);
                let at = (start * self.rate + burst.first * FACTOR as f64).round() as i64;
                if at + burst.samples.len() as i64 <= self.emitted {
                    self.late += 1;
                    return;
                }
                self.placed.push(Placed { at, samples: burst.samples });
            }
        }
    }

    /// Emits the next `count` output pairs: noise, plus every burst that
    /// overlaps them. False once the pipeline has gone.
    fn emit(&mut self, count: usize, running: &AtomicBool) -> bool {
        let sigma = self.boards.iter().map(|b| b.sigma).fold(f32::INFINITY, f32::min);
        if (sigma - self.noise_sigma).abs() > 0.05 * sigma {
            self.noise = self.unit_noise.iter().map(|&v| (v * sigma).round() as i16).collect();
            self.noise_sigma = sigma;
        }
        let from = self.emitted;
        let to = from + count as i64;
        let mut out = Vec::with_capacity(2 * count);
        while out.len() < 2 * count {
            let n = (2 * count - out.len()).min(self.noise.len() - self.noise_pos);
            out.extend_from_slice(&self.noise[self.noise_pos..self.noise_pos + n]);
            self.noise_pos = (self.noise_pos + n) % self.noise.len();
        }
        for p in &self.placed {
            let lo = p.at.max(from);
            let hi = (p.at + p.samples.len() as i64).min(to);
            if lo >= hi {
                continue;
            }
            let src = &p.samples[(lo - p.at) as usize..(hi - p.at) as usize];
            let dst = &mut out[2 * (lo - from) as usize..2 * (hi - from) as usize];
            for (d, v) in dst.chunks_exact_mut(2).zip(src) {
                d[0] = (d[0] as f32 + v.re) as i16;
                d[1] = (d[1] as f32 + v.im) as i16;
            }
        }
        self.placed.retain(|p| p.at + p.samples.len() as i64 > to);
        self.emitted = to;
        let mut chunk = out;
        loop {
            match self.tx.send_timeout(chunk, Duration::from_millis(200)) {
                Ok(()) => return true,
                Err(crossbeam::channel::SendTimeoutError::Timeout(c)) if running.load(Ordering::Relaxed) => chunk = c,
                Err(_) => return false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interfaces() {
        assert_eq!(boards_for("espdr0", 80_000_000).unwrap(), None);
        assert_eq!(boards_for("espdr", 16_000_000).unwrap(), None);
        assert!(boards_for("espdr", 40_000_000).is_err());
    }

    #[test]
    fn positions_are_dealt_round() {
        assert_eq!(position_mask(0, 1), 0);
        let masks: Vec<u16> = (0..5).map(|i| position_mask(i, 5)).collect();
        assert_eq!(masks.iter().fold(0, |a, m| a | m), 0xFFFF);
        assert!(masks.iter().all(|m| m.count_ones() >= 3));
        // The advertising channels around 2441 MHz (offsets k = 7, -1, -7)
        // fall to different boards.
        let owner = |k: i32| masks.iter().position(|m| m >> (k + 8) & 1 == 1).unwrap();
        assert_ne!(owner(7), owner(-1));
        assert_ne!(owner(7), owner(-7));
        assert_ne!(owner(-1), owner(-7));
    }

    #[test]
    fn folds_cover_what_the_esp_hears() {
        assert_eq!(Shaper::folds(7), vec![-25, -9, 7, 23, 39]);
        assert_eq!(Shaper::folds(-7), vec![-39, -23, -7, 9, 25]);
        assert_eq!(Shaper::folds(0), vec![-32, -16, 0, 16, 32]);
    }

    /// Power of `x[range]` at `mhz` (MHz at 80 Msps), relative to its
    /// total power.
    fn share_at(x: &[Complex32], range: std::ops::Range<usize>, mhz: f32) -> f32 {
        let mut c = Complex32::new(0.0, 0.0);
        let mut total = 0.0;
        for g in range.clone() {
            let a = -2.0 * std::f32::consts::PI * mhz * g as f32 / 80.0;
            c += x[g] * Complex32::new(a.cos(), a.sin());
            total += x[g].norm_sqr();
        }
        c.norm_sqr() / (range.len() as f32 * total)
    }

    #[test]
    fn a_channel_appears_at_each_fold() {
        // A steady signal 7 MHz below the LO in RF terms (LO-minus-RF
        // offset +7): 2434, 2402 (adv 37), 2418, 2450, 2466 around 2441.
        let shaper = Shaper::new();
        let pairs = vec![100u32; 1000]; // I = 100, Q = 0 at 4 Msps
        let s = shaper.narrow(1000, 7, &pairs);
        assert_eq!(s.samples.len(), (1000 + NARROW_TAPS_PER_PHASE) * 20);
        let mid = 2000..18000;
        for f in [-39.0, -23.0, -7.0, 9.0, 25.0] {
            let share = share_at(&s.samples, mid.clone(), f);
            assert!((share - 0.2).abs() < 0.01, "{} MHz holds {}", f, share);
        }
        // Each copy keeps the channel's level.
        let total: f32 = mid.clone().map(|g| s.samples[g].norm_sqr()).sum::<f32>() / mid.len() as f32;
        let level = (share_at(&s.samples, mid.clone(), -39.0) * total).sqrt();
        assert!((level - 6400.0).abs() < 100.0, "level {}", level);
        // Output 0 lies half the filter (139.5 outputs, 27.9 board pairs)
        // before the first channel output, centred on pair 997.5.
        assert!((s.first - (997.5 - 27.9)).abs() < 1e-9, "first {}", s.first);
    }

    #[test]
    fn merger_places_bursts_on_host_time() {
        let (tx, rx) = bounded::<Vec<i16>>(16);
        let t0 = Instant::now();
        let mut m = Merger::new(2, t0, tx);
        for b in &mut m.boards {
            b.sigma = 0.0;
        }
        // Board 1's pair 0 was taken at 1 ms; a burst whose output 0 is
        // board pair 160 then starts at 1 ms + 10 us = output 80800.
        m.boards[1].start = Some(0.001);
        m.boards[1].last_seen = 1.0; // no time for the estimate to creep
        let samples = vec![Complex32::new(1000.0, 0.0); 100];
        m.take(Event::Burst {
            board: 1,
            seen: t0 + Duration::from_secs(1),
            last: 0,
            burst: Shaped { first: 160.0, samples },
        });
        assert!(m.emit(1 << 17, &AtomicBool::new(true)));
        let out = rx.try_recv().unwrap();
        assert_eq!(out[2 * 80799], 0);
        assert_eq!(out[2 * 80800], 1000);
        assert_eq!(out[2 * 80899], 1000);
        assert_eq!(out[2 * 80900], 0);
        assert!(m.placed.is_empty());
    }
}
