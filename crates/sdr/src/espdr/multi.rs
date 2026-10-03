// Copyright 2025-2026 CEMAXECUTER LLC

//! One or more ESP32-S3 receivers as an 80 MHz receiver.
//!
//! With five boards (enough for 80 MHz) they are tiled: the baseband filter
//! is narrowed (see `FILTER_16M`) so each hears only its own 16 MHz, they
//! are tuned 16 MHz apart around the centre, staggered by half a MHz so no
//! Bluetooth channel lies on a boundary, and each burst is placed once
//! at its own frequency, which the receiver itself then gives. With fewer
//! boards they are folded: the filter is opened so that each hears the whole
//! band, as described below. `BD_ESPDR_LAYOUT` (`tile`, `tile-ble`, or
//! `fold`) and `BD_ESPDR_FILTER` override the choice. `tile-ble` tunes
//! the top board 1 MHz higher for stronger BLE advertising reception, at the
//! cost of one Classic channel; ordinary `tile` retains complete coverage.
//!
//! Folded:
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
//! the earliest arrivals relative to their last samples show when the
//! board's count began (within about a millisecond of USB scheduling). That
//! estimate is steered smoothly, with the crystals' drift, rather than
//! snapped to each new earliest arrival: a jump would move every burst after
//! it, and Classic Bluetooth's slot timing would not survive it. The
//! timeline runs `DELAY` behind real time so bursts still queued on an ESP
//! arrive before their place is emitted.
//!
//! A millisecond is too coarse for Classic Bluetooth, whose address and
//! clock recovery compare packets slot by slot (625 us), and consecutive
//! packets come from different boards. Current firmware therefore latches
//! its sample count at USB start-of-frame boundaries. Every board below one
//! host sees the same frame number; matching reports map each board onto the
//! reference's sample count and track its crystal drift to a few microseconds.
//!
//! Folded boards can refine that further: every board also sends one shared
//! position, where BLE advertising channel 38 folds, so the same bursts
//! arrive from every board. Their demodulated content gives each board's
//! offset from the reference to a fraction of a microsecond. Only one copy
//! of each shared burst is kept.
//!
//! The output timeline is the reference board's own sample count, scaled to
//! the output rate, so bursts keep exactly the spacing they were received
//! with, as they would from a single receiver. Host time only paces the
//! output.

use super::*;
use num_complex::Complex32;
use std::time::Instant;

/// Board sample rate.
const BOARD_RATE: f64 = 16e6;
/// Output sample rate, the span an ESP hears around its LO.
const FOLD_RATE: u32 = 80_000_000;
/// Frequency-placement unit. Half-MHz units let tiled LOs sit between the
/// integer-MHz Bluetooth channels while keeping mixer phases exact.
const HALF_MHZ_HZ: i64 = 500_000;
/// Output samples per board pair.
const FACTOR: usize = 5;
/// How far from the LO (MHz) a folded signal is still heard: measured flat
/// to about 25 MHz, 5 dB down at 39 and gone by 55.
const HEARING_MHZ: i64 = 40;
/// How far the output timeline runs behind real time.
const DELAY: f64 = 0.4;
/// Steering of a board's start estimate: for how long after its first
/// record it may jump to the earliest arrival seen, how often it is steered
/// after that (seconds), over how long a correction is spread, the share of
/// the error corrected and learned as drift each time, and the fastest it
/// may move (s/s). Steering only ever changes the estimate's rate.
const SETTLE: f64 = 1.5;
const STEER_EVERY: f64 = 0.5;
const STEER_SPREAD: f64 = 2.0;
const STEER_GAIN: f64 = 0.1;
const STEER_DRIFT_GAIN: f64 = 0.01;
const STEER_MAX_RATE: f64 = 100e-6;
/// Interpolation taps per output phase, for a 4 Msps channel and for a
/// 16 Msps window.
const NARROW_TAPS_PER_PHASE: usize = 14;
const WIDE_TAPS_PER_PHASE: usize = 48;
/// Output pairs per chunk handed to the pipeline.
const MULTI_CHUNK_PAIRS: usize = 1 << 18;
/// Complex samples in the background noise table.
const NOISE_PAIRS: usize = 1 << 18;
/// The frequency whose fold position every board shares for alignment: BLE
/// advertising channel 38, busy wherever there is BLE.
const SYNC_MHZ: i64 = 2426;
/// How far apart (seconds) a board's copy of a shared burst may be from the
/// reference's, while its offset is being found and once it is. Content
/// alone does not settle it at first: an advertiser repeats the same packet
/// every advertising interval, so a copy can match an earlier or later
/// broadcast. Offsets are therefore only taken once several agree.
const SYNC_WINDOW: f64 = 30e-3;
const SYNC_WINDOW_LOCKED: f64 = 0.3e-3;
/// Agreeing offsets (within SYNC_AGREE seconds) needed to lock a board on,
/// and by how many the winner must lead any other offset: a repeated
/// advertisement confirms an alias as often as the truth, and only other
/// packets tip the balance.
const SYNC_AGREEING: usize = 4;
const SYNC_LEAD: usize = 3;
const SYNC_AGREE: f64 = 80e-6;
/// Matches after which a board counts as aligned.
const SYNC_ALIGNED_AFTER: u32 = 10;
/// Content lags (4 Msps samples) searched between two copies' records, which
/// start where each board's detector happened to open them.
const SYNC_MAX_LAG: i64 = 160;
/// Correlation of the demodulated copies needed to call them one burst.
const SYNC_MIN_CORRELATION: f32 = 0.5;

/// How several boards share the band.
#[derive(Clone, Copy, PartialEq)]
enum Layout {
    /// Filter narrowed: each board hears its own 16 MHz; boards tuned 16 MHz
    /// apart.
    Tiled,
    /// As tiled, but move the upper board 1 MHz to keep BLE advertising
    /// channel 39 away from the analog filter's outer edge.
    TiledBle,
    /// Filter open: each board hears 80 MHz folded; boards on one LO, sharing
    /// out the channel positions.
    Folded,
}

impl Layout {
    fn tiled(self) -> bool {
        self != Self::Folded
    }
}

/// The layout for `count` boards, and the filter code it wants.
fn layout(count: usize) -> Result<(Layout, u16), String> {
    let tiles = (FOLD_RATE / 16_000_000) as usize;
    let layout = match std::env::var("BD_ESPDR_LAYOUT").as_deref() {
        Ok("tile") => Layout::Tiled,
        Ok("tile-ble") => Layout::TiledBle,
        Ok("fold") => Layout::Folded,
        Ok(other) => return Err(format!(
            "eSpDR: BD_ESPDR_LAYOUT must be tile, tile-ble, or fold, got '{}'", other
        )),
        Err(_) if count >= tiles => Layout::Tiled,
        Err(_) => Layout::Folded,
    };
    let filter = filter_code(if layout.tiled() { FILTER_16M } else { 0 })?;
    Ok((layout, filter))
}

/// Half-MHz units from the centre to board `index` of `count` when tiled.
/// The half-MHz stagger puts integer-MHz Bluetooth channels inside one tile
/// or the other rather than exactly on a 16 MHz boundary.
///
/// In the BLE-optimised layout the upper board moves 1 MHz so advertising
/// channel 39 is 1 MHz inside the analog filter rather than on its weak edge.
fn tile_offset_half_mhz(index: usize, count: usize, layout: Layout) -> i64 {
    let ble_top = layout == Layout::TiledBle && index + 1 == count;
    32 * index as i64 - 16 * (count as i64 - 1) - 1 + if ble_top { 2 } else { 0 }
}

/// Classic channels reported through an aliased position. Folded reception
/// is ambiguous everywhere; a tiled board is ambiguous only at its two
/// outermost half-MHz-staggered positions.
fn classic_alias_channels(layout: Layout, count: usize, center_freq: u64) -> Vec<u32> {
    if layout == Layout::Folded {
        return (2402..=2480).collect();
    }
    let mut channels = Vec::new();
    for index in 0..count {
        let lo = center_freq as i64
            + tile_offset_half_mhz(index, count, layout) * HALF_MHZ_HZ;
        for edge in [-7_500_000, 7_500_000] {
            let hz = lo + edge;
            if hz % 1_000_000 == 0 {
                let mhz = (hz / 1_000_000) as u32;
                if (2402..=2480).contains(&mhz) {
                    channels.push(mhz);
                }
            }
        }
    }
    channels.sort_unstable();
    channels.dedup();
    channels
}

/// The serial ports to use for `iface` at `sample_rate`, or None for a
/// single-board interface. `espdr` takes every attached ESP (up to five
/// when tiled); a comma list names them.
pub(super) fn boards_for(iface: &str, sample_rate: u32) -> Result<Option<Vec<String>>, String> {
    let list = iface.contains(',');
    if !list && (iface != "espdr" || sample_rate <= 16_000_000) {
        return Ok(None);
    }
    if sample_rate != FOLD_RATE {
        return Err(format!(
            "eSpDR: several ESPs make an 80 MHz receiver; use -C 80 with -i {} (got -C {})",
            iface,
            sample_rate / 1_000_000
        ));
    }
    let tiles = (FOLD_RATE / 16_000_000) as usize;
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
    let selected = layout(paths.len())?.0;
    let paths = if selected.tiled() && paths.len() > tiles {
        if list {
            return Err(format!("eSpDR: at most {} ESPs tile 80 MHz, {} were named", tiles, paths.len()));
        }
        paths.into_iter().take(tiles).collect()
    } else {
        paths
    };
    if selected == Layout::TiledBle && paths.len() != tiles {
        return Err(format!(
            "eSpDR: BD_ESPDR_LAYOUT=tile-ble needs exactly {} boards, got {}",
            tiles,
            paths.len()
        ));
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
/// to different boards, plus the shared position `sync` (offset k) if any.
fn position_mask(index: usize, count: usize, sync: Option<i32>) -> u16 {
    if count == 1 {
        return 0;
    }
    let own = (0..16).filter(|bit| bit % count == index).fold(0u16, |m, bit| m | 1 << bit);
    own | sync.map_or(0, |k| 1 << (k + 8))
}

/// The channel offset k (LO-minus-RF, MHz) at which `mhz` appears for an LO
/// of `lo_mhz`, if the firmware can report it (-7..7).
fn fold_position(mhz: i64, lo_mhz: i64) -> Option<i32> {
    let rf = (mhz - lo_mhz + 8).rem_euclid(16) - 8; // RF-minus-LO, -8..7
    let k = -rf as i32;
    (-7..=7).contains(&k).then_some(k)
}

/// The board whose own positions include offset `k`.
fn owner(k: i32, count: usize) -> usize {
    (k + 8) as usize % count
}

pub(super) fn open(paths: &[String], center_freq: u64, gain: i32) -> Result<EspdrHandle, String> {
    let count = paths.len();
    let gains = board_gains(count, gain)?;
    let (stream_arg, channelize, reject) = stream_arg()?;
    let t0 = Instant::now();
    let (event_tx, event_rx) = bounded::<Event>(4096);
    let running = Arc::new(AtomicBool::new(true));
    let mut board_gain_tx = Vec::with_capacity(count);
    let mut threads = Vec::with_capacity(count);
    let mut lo_hz = center_freq as u32;
    let (layout, filter) = layout(count)?;
    let classic_alias_channels = classic_alias_channels(layout, count, center_freq);
    let folded = layout == Layout::Folded;
    if layout == Layout::TiledBle && count != (FOLD_RATE / 16_000_000) as usize {
        return Err(format!("eSpDR: BLE tiling needs exactly 5 boards, got {}", count));
    }
    if layout == Layout::TiledBle && center_freq != 2_441_000_000 {
        return Err(format!(
            "eSpDR: BD_ESPDR_LAYOUT=tile-ble covers Bluetooth at -c 2441, got {:.3} MHz",
            center_freq as f64 / 1e6
        ));
    }
    if layout == Layout::TiledBle {
        eprintln!("eSpDR: BLE tiling enabled; 2465 MHz Classic channel 63 is not received");
    }
    // Whole-window bursts cannot be shared by position, nor aligned by one;
    // tiled boards hear nothing in common.
    let sync = if folded && channelize && count > 1 {
        fold_position(SYNC_MHZ, (center_freq as i64 + 500_000) / 1_000_000)
    } else {
        None
    };
    // Set every board up before any starts streaming, and start them all
    // before any is handed to a thread, so that a board that fails leaves
    // none of the others streaming with no one to read or stop it.
    let mut boards = Vec::with_capacity(count);
    for (index, path) in paths.iter().enumerate() {
        let offset = if folded { 0 } else { tile_offset_half_mhz(index, count, layout) };
        let lo = (center_freq as i64 + offset * HALF_MHZ_HZ) as u32;
        let (link, tuned) = open_board(path, 1, 20, gains[index], lo, filter)?;
        if offset == 0 {
            lo_hz = tuned;
        }
        let mask = if folded && channelize { position_mask(index, count, sync) } else { 0 };
        boards.push((index, path, link, offset, mask, tuned));
    }
    for started in 0..boards.len() {
        let (_, path, link, _, mask, _) = &mut boards[started];
        let failure = match link.start_stream_masked(stream_arg, *mask) {
            Ok(true) => None,
            Ok(false) => Some(format!("eSpDR: the ESP at {} cannot stream; load the current firmware", path)),
            Err(e) => Some(e),
        };
        if let Some(e) = failure {
            for (_, _, link, _, _, _) in &mut boards[..started] {
                link.stop_stream();
            }
            return Err(e);
        }
    }
    for (index, path, link, offset, mask, tuned) in boards {
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
        let shaper = Shaper::new(if folded { None } else { Some(offset) });
        threads.push(
            std::thread::Builder::new()
                .name(format!("espdr-board{}", index))
                .spawn(move || board_loop(index, link, shaper, events, gain_rx, &running, stream_arg, mask))
                .map_err(|e| format!("eSpDR: cannot start receive thread: {}", e))?,
        );
    }
    drop(event_tx);
    eprintln!(
        "eSpDR: {} board{} {} {:.3} MHz, {}",
        count,
        if count == 1 { "" } else { "s" },
        if folded { "hearing 80 MHz folded into 16 around" } else { "tiling 16 MHz each around" },
        center_freq as f64 / 1e6,
        match (channelize, reject) {
            (true, true) => "channelized bursts (Wi-Fi rejected)",
            (true, false) => "channelized bursts",
            (false, true) => "bursts (Wi-Fi rejected)",
            (false, false) => "bursts",
        }
    );
    if folded && !channelize && count > 1 {
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
                merger.sync = sync.map(|k| (k, owner(k, count)));
                if let Some((_, reference)) = merger.sync {
                    merger.reference = Some(reference);
                }
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
        classic_alias_channels,
    })
}

enum Event {
    /// A burst ready for the timeline, from a record received at `seen`
    /// whose last pair was board pair `last`.
    Burst { board: usize, seen: Instant, last: u64, burst: Shaped },
    Status { board: usize, seen: Instant, at: u64, usb_frame: Option<u16>, words: Vec<u32> },
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
    let mut telemetry = ReceiverTelemetry::default();
    let mut telemetry_reported = Instant::now();
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
            Ok(Record::Status { start, usb_frame, words }) => {
                telemetry.add(&words);
                if words.len() >= STATUS_V2_WORDS
                    && telemetry_reported.elapsed() >= Duration::from_secs(1)
                {
                    telemetry.report(Some(board));
                    telemetry_reported = Instant::now();
                }
                Event::Status { board, seen: Instant::now(), at: start, usb_frame, words }
            }
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
    /// A channelized burst's offset (LO-minus-RF, MHz), and its channel at
    /// 4 Msps (output j at output sample `NARROW_DELAY + 20 j` of `samples`).
    offset: Option<i32>,
    channel: Option<Vec<Complex32>>,
}

/// Interpolates bursts to the output rate and places them at the frequency
/// they came from: for a tiled board, its own; for a folded one, each it
/// could have come from.
struct Shaper {
    /// A tiled board's LO, in half-MHz units from the centre; None when
    /// folded.
    tile: Option<i64>,
    /// e^(j 2 pi m / 160): half-MHz offsets at 80 Msps repeat every 160
    /// samples.
    turn: Vec<Complex32>,
    /// From a 4 Msps channel: 20 phases of NARROW_TAPS_PER_PHASE.
    narrow_taps: Vec<Vec<f32>>,
    /// From the 16 Msps window: 5 phases of WIDE_TAPS_PER_PHASE.
    wide_taps: Vec<Vec<f32>>,
}

impl Shaper {
    fn new(tile: Option<i64>) -> Self {
        let period = (FOLD_RATE as i64 / HALF_MHZ_HZ) as usize;
        let turn = (0..period)
            .map(|m| {
                let a = 2.0 * std::f64::consts::PI * m as f64 / period as f64;
                Complex32::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let rate = FOLD_RATE as f64 / 1e6; // output rate in MHz
        // A channel's content lies within +-1.6 MHz and its first image
        // starts 2.4 MHz out; a window's lies within +-8 MHz, imaged from
        // 8 MHz on, so keep it to about +-7.4.
        let narrow_taps = polyphase(lowpass(4 * FACTOR, NARROW_TAPS_PER_PHASE, 2.0 / rate, 4.5), 4 * FACTOR);
        let wide_taps = polyphase(lowpass(FACTOR, WIDE_TAPS_PER_PHASE, 8.0 / rate, 5.65), FACTOR);
        Self { tile, turn, narrow_taps, wide_taps }
    }

    /// Where (half-MHz units from the centre) a signal seen `offset` MHz
    /// above the board's LO goes: tiled, just there; folded, everywhere it
    /// could have come from.
    fn places(&self, offset: i64) -> Vec<i64> {
        match self.tile {
            Some(lo) => vec![lo + 2 * offset],
            None => Self::folds(offset).into_iter().map(|f| 2 * f).collect(),
        }
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
        let mut raw: Vec<(f32, f32)> = pairs
            .iter()
            .map(|&w| (((w & 1023) ^ 512) as f32 - 512.0, (((w >> 10) & 1023) ^ 512) as f32 - 512.0))
            .collect();
        if self.tile.is_some() {
            remove_dc(&mut raw, offset);
        }
        // Conjugate to RF-minus-LO and scale like convert_pairs.
        let x: Vec<Complex32> = raw.iter().map(|&(i, q)| Complex32::new(i, -q) * SAMPLE_SCALE as f32).collect();
        let step = 1.0 / FACTOR as f64; // board pairs per output sample
        let delay = (self.narrow_taps.len() * NARROW_TAPS_PER_PHASE - 1) as f64 / 2.0;
        Shaped {
            first: start as f64 - 2.5 - delay * step,
            samples: self.interpolate(&x, &self.narrow_taps, &self.places(-offset as i64)),
            offset: Some(offset),
            channel: Some(x),
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
            samples: self.interpolate(&x, &self.wide_taps, &self.places(0)),
            offset: None,
            channel: None,
        }
    }

    /// Interpolates `x` by the number of phases and repeats it at each of
    /// `offsets` (half-MHz units).
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

/// The lag (4 Msps samples, fractional) at which `other` holds the same
/// signal as `reference`: `other[j + lag]` is `reference[j]`. Found by
/// correlating their instantaneous frequency, which the two receivers'
/// differing phase, gain and small frequency offset do not disturb; None
/// unless the correlation shows them to be one burst.
fn content_lag(reference: &[Complex32], other: &[Complex32]) -> Option<f64> {
    let demod = |x: &[Complex32]| -> Vec<f32> {
        let d: Vec<f32> = x.windows(2).map(|w| (w[1] * w[0].conj()).arg()).collect();
        let mean = d.iter().sum::<f32>() / d.len().max(1) as f32;
        d.into_iter().map(|v| v - mean).collect()
    };
    let (a, b) = (demod(reference), demod(other));
    let score = |lag: i64| -> f32 {
        let (mut ab, mut aa, mut bb) = (0.0f32, 0.0f32, 0.0f32);
        let mut n = 0;
        for (j, &x) in a.iter().enumerate() {
            let k = j as i64 + lag;
            if k < 0 || k >= b.len() as i64 {
                continue;
            }
            let y = b[k as usize];
            ab += x * y;
            aa += x * x;
            bb += y * y;
            n += 1;
        }
        if n < 200 || aa <= 0.0 || bb <= 0.0 {
            return 0.0;
        }
        ab / (aa * bb).sqrt()
    };
    let scores: Vec<(i64, f32)> = (-SYNC_MAX_LAG..=SYNC_MAX_LAG).map(|l| (l, score(l))).collect();
    let &(best, peak) = scores.iter().max_by(|x, y| x.1.total_cmp(&y.1))?;
    if peak < SYNC_MIN_CORRELATION {
        return None;
    }
    // Refine between samples with a parabola through the peak.
    let at = |l: i64| scores.iter().find(|s| s.0 == l).map_or(0.0, |s| s.1);
    let (l, c, r) = (at(best - 1), peak, at(best + 1));
    let curve = l - 2.0 * c + r;
    let shift = if curve < 0.0 { (0.5 * (l - r) / curve).clamp(-0.5, 0.5) } else { 0.0 };
    Some(best as f64 + shift as f64)
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
    /// Host time (seconds since start) at which the board's pair 0 was taken,
    /// as estimated at host time `steered`; the rate it moves at until the
    /// next steering (s/s), and the crystals' drift learned so far.
    start: Option<f64>,
    steered: f64,
    rate: f64,
    drift: f64,
    /// Host time of the board's first record, and the earliest start shown
    /// by arrivals since `window` (host time).
    first_seen: f64,
    earliest: f64,
    window: f64,
    /// Background noise per component, in output units.
    sigma: f32,
    /// Bursts dropped and blocks skipped on the ESP, from its last status.
    lost: u64,
    /// Once measured against the reference board: the reference's pair
    /// count less this board's (pairs) at this board's pair `anchor`, how
    /// that changes per pair (the crystals' drift), the anchor, and how many
    /// shared bursts it has been measured from.
    relative: Option<(f64, f64, f64)>,
    matches: u32,
    /// While unlocked, the offsets (pairs) measured so far, each with the
    /// reference burst (its output sample) it was measured against.
    candidates: Vec<(f64, i64)>,
    /// Recent USB SOF latches, and matched (own pair, reference-minus-own
    /// pair) measurements used by tiled receivers.
    usb_stamps: Vec<UsbStamp>,
    usb_samples: Vec<(f64, f64)>,
}

#[derive(Clone, Copy)]
struct UsbStamp {
    frame: u16,
    pair: f64,
    seen: f64,
}

/// A burst placed on the output timeline.
struct Placed {
    at: i64,
    samples: Vec<Complex32>,
    board: usize,
    /// The board pair (fractional) of its output sample 0.
    first: f64,
    /// For a burst at the shared position, its channel at 4 Msps (see
    /// `Shaped::channel`), for alignment.
    sync: Option<Vec<Complex32>>,
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
    /// The shared position (offset k) and the reference board, if aligning.
    sync: Option<(i32, usize)>,
    /// The board whose sample count is the output timeline (the sync
    /// reference, or the only board), and the output sample of its pair 0.
    reference: Option<usize>,
    origin: Option<f64>,
}

impl Merger {
    fn new(count: usize, t0: Instant, tx: Sender<Vec<i16>>) -> Self {
        let boards = (0..count)
            .map(|_| Board {
                start: None,
                steered: 0.0,
                rate: 0.0,
                drift: 0.0,
                first_seen: 0.0,
                earliest: f64::INFINITY,
                window: 0.0,
                sigma: 4.0 * SAMPLE_SCALE as f32,
                lost: 0,
                relative: None,
                matches: 0,
                candidates: Vec::new(),
                usb_stamps: Vec::new(),
                usb_samples: Vec::new(),
            })
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
            sync: None,
            reference: (count == 1).then_some(0),
            origin: None,
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
            let target = match (self.reference, self.origin) {
                // The reference's pair DELAY ago, on the timeline.
                (Some(r), Some(origin)) if self.boards[r].start.is_some() => {
                    (origin + (now - DELAY - self.own_start(r, now)) * BOARD_RATE * FACTOR as f64) as i64
                }
                _ => ((now - DELAY) * self.rate) as i64,
            };
            if target - self.emitted >= MULTI_CHUNK_PAIRS as i64 && !self.emit(MULTI_CHUNK_PAIRS, running) {
                return;
            }
        }
    }

    /// Updates the board's start estimate from a record's arrival (host
    /// time `seen`, last pair `last`).
    fn track(&mut self, board: usize, seen: Instant, last: u64) {
        let seen = seen.duration_since(self.t0).as_secs_f64();
        let b = &mut self.boards[board];
        let candidate = seen - last as f64 / BOARD_RATE;
        let Some(start) = b.start else {
            b.start = Some(candidate);
            b.steered = seen;
            b.first_seen = seen;
            b.window = seen;
            b.earliest = f64::INFINITY;
            return;
        };
        b.earliest = b.earliest.min(candidate);
        if seen - b.window < STEER_EVERY {
            return;
        }
        let now = start + b.rate * (seen - b.steered);
        let error = b.earliest - now;
        if seen - b.first_seen < SETTLE {
            b.start = Some(now.min(b.earliest));
        } else {
            // Carry on from where the estimate is (no jump) and correct by
            // moving at a slightly different rate for a while.
            b.drift += STEER_DRIFT_GAIN * error / (seen - b.steered);
            b.start = Some(now);
            b.rate = (b.drift + STEER_GAIN * error / STEER_SPREAD).clamp(-STEER_MAX_RATE, STEER_MAX_RATE);
        }
        b.steered = seen;
        b.window = seen;
        b.earliest = f64::INFINITY;
    }

    /// A board's own start estimate at host time `now`.
    fn own_start(&self, board: usize, now: f64) -> f64 {
        let b = &self.boards[board];
        b.start.map_or(0.0, |s| s + b.rate * (now - b.steered))
    }

    /// Matches one board's sample count to the reference at a USB SOF. The
    /// firmware reports only common 256-frame boundaries, and arrival time
    /// distinguishes the same 11-bit frame number after its 2.048 s wrap.
    fn usb_stamp(&mut self, board: usize, frame: u16, pair: u64, seen: Instant) {
        // Leave older firmware on the existing host-time path. Seeing the
        // extension opts this stream into board 0's sample timeline.
        let reference = *self.reference.get_or_insert(0);
        let stamp = UsbStamp { frame, pair: pair as f64, seen: seen.duration_since(self.t0).as_secs_f64() };
        let stamps = &mut self.boards[board].usb_stamps;
        stamps.push(stamp);
        if stamps.len() > 16 {
            stamps.remove(0);
        }
        let nearest = |stamps: &[UsbStamp], target: UsbStamp| {
            stamps
                .iter()
                .copied()
                .filter(|s| s.frame == target.frame && (s.seen - target.seen).abs() < 0.5)
                .min_by(|a, b| (a.seen - target.seen).abs().total_cmp(&(b.seen - target.seen).abs()))
        };
        if board == reference {
            let matches: Vec<(usize, UsbStamp)> = (0..self.boards.len())
                .filter(|&b| b != reference)
                .filter_map(|b| nearest(&self.boards[b].usb_stamps, stamp).map(|s| (b, s)))
                .collect();
            for (b, other) in matches {
                self.measure_usb(b, other.pair, stamp.pair);
            }
        } else if let Some(r) = nearest(&self.boards[reference].usb_stamps, stamp) {
            self.measure_usb(board, stamp.pair, r.pair);
        }
    }

    /// Fits reference-minus-own pair offset and drift to recent USB SOF
    /// matches. Regression averages the measured 1--2 us latch jitter while
    /// retaining the boards' several-ppm crystal differences.
    fn measure_usb(&mut self, board: usize, pair: f64, reference_pair: f64) {
        let b = &mut self.boards[board];
        if b.usb_samples.last().is_some_and(|&(p, _)| p == pair) {
            return;
        }
        b.usb_samples.push((pair, reference_pair - pair));
        if b.usb_samples.len() > 32 {
            b.usb_samples.remove(0);
        }
        let n = b.usb_samples.len() as f64;
        let mean_x = b.usb_samples.iter().map(|s| s.0).sum::<f64>() / n;
        let mean_y = b.usb_samples.iter().map(|s| s.1).sum::<f64>() / n;
        let variance = b.usb_samples.iter().map(|s| (s.0 - mean_x).powi(2)).sum::<f64>();
        let drift = if variance > 0.0 {
            b.usb_samples.iter().map(|s| (s.0 - mean_x) * (s.1 - mean_y)).sum::<f64>() / variance
        } else {
            0.0
        };
        let offset = mean_y + drift * (pair - mean_x);
        b.relative = Some((offset, drift, pair));
        b.matches = b.usb_samples.len() as u32;
        if b.matches == 3 {
            eprintln!("eSpDR: board {} aligned to USB frame clock", board);
        }
    }

    fn take(&mut self, event: Event) {
        match event {
            Event::Restart { board } => {
                // Its count starts again; so does its alignment (everyone's,
                // and the timeline's origin, if it is the reference).
                self.boards[board].start = None;
                let all = self.reference == Some(board);
                if all {
                    self.origin = None;
                }
                for (i, b) in self.boards.iter_mut().enumerate() {
                    if all || i == board {
                        b.relative = None;
                        b.matches = 0;
                        b.candidates.clear();
                        b.usb_stamps.clear();
                        b.usb_samples.clear();
                    }
                }
            }
            Event::Status { board, seen, at, usb_frame, words } => {
                self.track(board, seen, at);
                if self.sync.is_none() {
                    if let Some(frame) = usb_frame {
                        self.usb_stamp(board, frame, at, seen);
                    }
                }
                let b = &mut self.boards[board];
                let floor = f32::from_bits(words[0]);
                if floor > 0.0 {
                    b.sigma = (floor / 2.0).sqrt() * SAMPLE_SCALE as f32;
                }
                b.lost = words[3] as u64 + words[5] as u64;
            }
            Event::Burst { board, seen, last, burst } => {
                self.track(board, seen, last);
                let now = seen.duration_since(self.t0).as_secs_f64();
                let at = self.position(board, burst.first, now);
                if at + burst.samples.len() as i64 <= self.emitted {
                    self.late += 1;
                    return;
                }
                let sync = match (self.sync, burst.offset, burst.channel) {
                    (Some((k, _)), Some(o), Some(channel)) if o == k => Some(channel),
                    _ => None,
                };
                if let Some(channel) = &sync {
                    if self.align(board, at, burst.first, channel, now) {
                        return; // another board's copy of a burst already placed
                    }
                }
                self.placed.push(Placed { at, samples: burst.samples, board, first: burst.first, sync });
            }
        }
    }

    /// Aligns boards on a burst at the shared position, from `board` placed
    /// at output sample `at`, with its channel at 4 Msps. A copy from the
    /// reference board measures the other boards' copies already placed,
    /// and replaces those of locked boards; another board's copy is measured
    /// against the reference's, and dropped (true) if its board is locked.
    fn align(&mut self, board: usize, at: i64, first: f64, channel: &[Complex32], now: f64) -> bool {
        let Some((_, reference)) = self.sync else { return false };
        let rate = self.rate;
        let window = |b: &Board| {
            (if b.relative.is_some() { SYNC_WINDOW_LOCKED } else { SYNC_WINDOW } * rate) as i64
        };
        // Output samples by which `other` (placed at `other_at`) runs ahead
        // of the reference copy, if the two hold the same packet.
        let offset = |window: i64, reference_at: i64, reference: &[Complex32], other_at: i64, other: &[Complex32]| {
            if (other_at - reference_at).abs() > window {
                return None;
            }
            let lag = content_lag(reference, other)?;
            let delta = (other_at - reference_at) as f64 + (4 * FACTOR) as f64 * lag;
            (delta.abs() <= window as f64).then_some(delta)
        };
        if board == reference {
            let mut measured = Vec::new();
            let boards = &self.boards;
            self.placed.retain(|p| match &p.sync {
                Some(other) if p.board != reference => {
                    match offset(window(&boards[p.board]), at, channel, p.at, other) {
                        Some(delta) => {
                            measured.push((p.board, p.first, vec![(delta, at)]));
                            boards[p.board].relative.is_none() // keep copies until locked
                        }
                        None => true,
                    }
                }
                _ => true,
            });
            for (b, first, matches) in measured {
                self.measure(b, first, &matches, now);
            }
            false
        } else {
            let w = window(&self.boards[board]);
            let matches: Vec<(f64, i64)> = self
                .placed
                .iter()
                .filter(|p| p.board == reference)
                .filter_map(|p| Some((offset(w, p.at, p.sync.as_deref()?, at, channel)?, p.at)))
                .collect();
            if matches.is_empty() {
                return false;
            }
            let locked = self.boards[board].relative.is_some();
            self.measure(board, first, &matches, now);
            locked
        }
    }

    /// Takes in the offsets measured for a copy from `board` whose output 0
    /// is its pair `first` (output samples by which it runs ahead of each
    /// reference burst it matched), at host time `now`. A locked board is
    /// corrected by the nearest. Otherwise they are candidates, and the board
    /// locks on once one offset is confirmed by at least SYNC_AGREEING
    /// different reference bursts and by SYNC_LEAD more than any other
    /// (repeats of one advertisement also match whole advertising intervals
    /// off; only other packets settle which).
    fn measure(&mut self, board: usize, first: f64, matches: &[(f64, i64)], now: f64) {
        if self.boards[board].relative.is_some() {
            if let Some(&(delta, _)) = matches.iter().min_by(|a, b| a.0.abs().total_cmp(&b.0.abs())) {
                self.adjust(board, first, delta);
            }
            return;
        }
        // The offset (pairs) between the board's samples and the
        // reference's by the coarse host-time mapping, less what was
        // measured.
        let base = self.reference_pair(board, first, now) - first;
        let agree = SYNC_AGREE * BOARD_RATE;
        let b = &mut self.boards[board];
        for &(delta, reference_at) in matches {
            b.candidates.push((base - delta / FACTOR as f64, reference_at));
        }
        // Each distinct offset, with how many reference bursts confirm it.
        let mut clusters: Vec<(usize, Vec<f64>)> = Vec::new();
        for &(c, _) in &b.candidates {
            let cluster: Vec<&(f64, i64)> = b.candidates.iter().filter(|d| (d.0 - c).abs() <= agree).collect();
            let mut bursts: Vec<i64> = cluster.iter().map(|d| d.1).collect();
            bursts.sort_unstable();
            bursts.dedup();
            clusters.push((bursts.len(), cluster.iter().map(|d| d.0).collect()));
        }
        clusters.sort_by(|x, y| y.0.cmp(&x.0));
        let best = clusters.first().cloned();
        let centre = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let rival = best.as_ref().and_then(|(_, top)| {
            clusters
                .iter()
                .find(|(_, v)| (centre(v) - centre(top)).abs() > agree)
                .map(|(support, _)| *support)
        });
        let best = best.filter(|(support, _)| *support >= SYNC_AGREEING && *support >= rival.unwrap_or(0) + SYNC_LEAD);
        if let Some((_, mut values)) = best {
            values.sort_by(f64::total_cmp);
            b.relative = Some((values[values.len() / 2], 0.0, first));
            b.matches = SYNC_AGREEING as u32;
            b.candidates.clear();
        } else if b.candidates.len() > 256 {
            b.candidates.drain(..64);
        }
    }

    /// The reference board's pair (fractional) taken at the same moment as
    /// `board`'s pair `p`: by the measured offset once locked, else by the
    /// boards' host-time estimates at host time `now`.
    fn reference_pair(&self, board: usize, p: f64, now: f64) -> f64 {
        let Some(r) = self.reference else { return p };
        if board == r {
            return p;
        }
        match self.boards[board].relative {
            Some((offset, drift, anchor)) => p + offset + drift * (p - anchor),
            None => p + (self.own_start(board, now) - self.own_start(r, now)) * BOARD_RATE,
        }
    }

    /// The output sample at which `board`'s pair `p` goes: on the reference
    /// board's timeline once there is one, else by host time.
    fn position(&mut self, board: usize, p: f64, now: f64) -> i64 {
        if let Some(r) = self.reference.filter(|&r| self.boards[r].start.is_some()) {
            let start = self.own_start(r, now);
            let origin = *self.origin.get_or_insert(start * self.rate);
            return (origin + self.reference_pair(board, p, now) * FACTOR as f64).round() as i64;
        }
        (self.own_start(board, now) * self.rate + p * FACTOR as f64).round() as i64
    }

    /// Corrects a locked board's offset and drift by a measured error
    /// (output samples ahead of the reference) on a copy whose output 0 is
    /// its pair `first`; each measurement is good to a fraction of a
    /// microsecond.
    fn adjust(&mut self, board: usize, first: f64, delta: f64) {
        let error = delta / FACTOR as f64; // pairs
        let b = &mut self.boards[board];
        if let Some((offset, drift, anchor)) = b.relative {
            let gain = if b.matches < SYNC_ALIGNED_AFTER { 0.5 } else { 0.3 };
            let current = offset + drift * (first - anchor);
            let span = (first - anchor).max(0.1 * BOARD_RATE);
            b.relative = Some((current - gain * error, drift - 0.05 * error / span, first));
            b.matches += 1;
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
        assert_eq!(position_mask(0, 1, None), 0);
        let masks: Vec<u16> = (0..5).map(|i| position_mask(i, 5, None)).collect();
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
        let shaper = Shaper::new(None);
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
        m.boards[1].window = 1.0; // not due to be steered at the arrival below
        let samples = vec![Complex32::new(1000.0, 0.0); 100];
        m.take(Event::Burst {
            board: 1,
            seen: t0 + Duration::from_secs(1),
            last: 0,
            burst: Shaped { first: 160.0, samples, offset: None, channel: None },
        });
        assert!(m.emit(1 << 17, &AtomicBool::new(true)));
        let out = rx.try_recv().unwrap();
        assert_eq!(out[2 * 80799], 0);
        assert_eq!(out[2 * 80800], 1000);
        assert_eq!(out[2 * 80899], 1000);
        assert_eq!(out[2 * 80900], 0);
        assert!(m.placed.is_empty());
    }

    #[test]
    fn adv38_folds_to_a_shared_position() {
        assert_eq!(fold_position(2426, 2441), Some(-1));
        assert_eq!(fold_position(2402, 2441), Some(7));
        assert_eq!(fold_position(2480, 2441), Some(-7));
        assert_eq!(fold_position(2426, 2426), Some(0));
        assert_eq!(fold_position(2434, 2426), None); // 8 MHz up: not reported
        let masks: Vec<u16> = (0..5).map(|i| position_mask(i, 5, Some(-1))).collect();
        assert!(masks.iter().all(|m| m >> 7 & 1 == 1));
        assert_eq!(owner(-1, 5), 2);
    }

    #[test]
    fn classic_alias_positions_follow_the_layout() {
        assert_eq!(
            classic_alias_channels(Layout::Tiled, 5, 2_441_000_000),
            vec![2416, 2417, 2432, 2433, 2448, 2449, 2464, 2465, 2480]
        );
        assert_eq!(
            classic_alias_channels(Layout::TiledBle, 5, 2_441_000_000),
            vec![2416, 2417, 2432, 2433, 2448, 2449, 2464, 2466]
        );
        let folded = classic_alias_channels(Layout::Folded, 2, 2_441_000_000);
        assert_eq!(folded.len(), 79);
        assert_eq!(folded.first(), Some(&2402));
        assert_eq!(folded.last(), Some(&2480));
    }

    /// A GFSK-like burst: random bits at 1 Mbit/s, 4 samples per bit,
    /// +-250 kHz, with a phase and frequency offset, `pad` samples of noise
    /// first, and a little noise throughout.
    fn burst(bits: &[u8], pad: usize, phase: f32, cfo_hz: f32, seed: u32) -> Vec<Complex32> {
        let mut state = seed;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32 - 0.5) * 20.0
        };
        let mut out = Vec::new();
        for _ in 0..pad {
            out.push(Complex32::new(noise(), noise()));
        }
        let mut ph = phase;
        for &bit in bits {
            for _ in 0..4 {
                let f = if bit == 1 { 250e3 } else { -250e3 } + cfo_hz;
                ph += 2.0 * std::f32::consts::PI * f / 4e6;
                out.push(Complex32::new(1000.0 * ph.cos() + noise(), 1000.0 * ph.sin() + noise()));
            }
        }
        out
    }

    #[test]
    fn copies_are_recognised_by_content() {
        let mut state = 7u32;
        let bits: Vec<u8> = (0..400)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8 & 1
            })
            .collect();
        let reference = burst(&bits, 40, 0.3, 1000.0, 1);
        // The same packet opened 23 samples later by another receiver.
        let other = burst(&bits, 63, 2.0, -3000.0, 2);
        let lag = content_lag(&reference, &other).unwrap();
        assert!((lag - 23.0).abs() < 0.3, "lag {}", lag);
        // A different packet is not taken for a copy.
        let different: Vec<u8> = bits.iter().rev().copied().collect();
        assert!(content_lag(&reference, &burst(&different, 40, 0.0, 0.0, 3)).is_none());
    }

    #[test]
    fn boards_are_pulled_onto_the_reference() {
        let (tx, _rx) = bounded::<Vec<i16>>(16);
        let t0 = Instant::now();
        let mut m = Merger::new(3, t0, tx);
        m.sync = Some((-1, 0));
        m.reference = Some(0);
        for b in &mut m.boards {
            b.start = Some(1.0);
            b.window = 2.0; // not due to be steered at the arrivals below
        }
        // Board 2's clock reads 4 ms late: its copies land 320000 samples
        // after the reference's. Each copy is a different packet.
        m.boards[2].start = Some(1.004);
        let mut state = 99u32;
        for n in 0..30 {
            let bits: Vec<u8> = (0..300)
                .map(|_| {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    (state >> 16) as u8 & 1
                })
                .collect();
            let first = 1_000_000.0 * (n + 1) as f64;
            let seen = t0 + Duration::from_secs(2);
            let copy = |pad: usize, seed: u32| Shaped {
                first: first - 4.0 * pad as f64, // opened `pad` channel samples (4 board pairs each) early
                samples: vec![Complex32::new(1.0, 0.0); 4000],
                offset: Some(-1),
                channel: Some(burst(&bits, pad, seed as f32, 500.0 * seed as f32, seed)),
            };
            m.take(Event::Burst { board: 0, seen, last: 0, burst: copy(40, 1) });
            m.take(Event::Burst { board: 2, seen, last: 0, burst: copy(40 + (n % 7) as usize, 2) });
        }
        // Aligned to well under a microsecond, and only the reference's
        // copies kept once locked on (the first few are kept from both).
        // The same burst carries the same pair number on both boards, so
        // board 2's samples must map onto the reference's own.
        let offset = m.reference_pair(2, 1e7, 2.0) - 1e7;
        assert!(offset.abs() < 8.0, "offset {} pairs", offset);
        assert!(m.boards[2].matches >= SYNC_ALIGNED_AFTER);
        assert_eq!(m.placed.iter().filter(|p| p.board == 0).count(), 30);
        assert!(m.placed.iter().filter(|p| p.board == 2).count() <= SYNC_AGREEING);
    }

    #[test]
    fn repeated_advertisements_do_not_mislead() {
        // One advertiser repeats the same packet every 20 ms; board 2 reads
        // 13 ms late, so its copies sit nearer the next broadcast (7 ms)
        // than their own. Other advertisers' packets now and then settle it.
        let (tx, _rx) = bounded::<Vec<i16>>(16);
        let t0 = Instant::now();
        let mut m = Merger::new(3, t0, tx);
        m.sync = Some((-1, 0));
        m.reference = Some(0);
        for b in &mut m.boards {
            b.start = Some(1.0);
            b.window = 2.0; // not due to be steered at the arrivals below
        }
        m.boards[2].start = Some(1.013);
        let seen = t0 + Duration::from_secs(2);
        let mut state = 5u32;
        let mut bits = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    (state >> 16) as u8 & 1
                })
                .collect()
        };
        let beacon = bits(300);
        let copy = |first: f64, b: &[u8], seed: u32| Shaped {
            first,
            samples: vec![Complex32::new(1.0, 0.0); 4000],
            offset: Some(-1),
            channel: Some(burst(b, 40, seed as f32, 0.0, seed)),
        };
        for n in 0..40 {
            let first = 320_000.0 * (n + 1) as f64; // every 20 ms
            let content = if n % 5 == 4 { bits(300) } else { beacon.clone() };
            // Either copy may arrive first.
            let (a, b) = (copy(first, &content, 1), copy(first, &content, 2));
            if n % 2 == 0 {
                m.take(Event::Burst { board: 0, seen, last: 0, burst: a });
                m.take(Event::Burst { board: 2, seen, last: 0, burst: b });
            } else {
                m.take(Event::Burst { board: 2, seen, last: 0, burst: b });
                m.take(Event::Burst { board: 0, seen, last: 0, burst: a });
            }
        }
        let offset = m.reference_pair(2, 1e7, 2.0) - 1e7;
        assert!(offset.abs() < 16.0, "offset {} pairs", offset);
    }

    #[test]
    fn start_estimate_is_steered_not_snapped() {
        let (tx, _rx) = bounded::<Vec<i16>>(16);
        let t0 = Instant::now();
        let mut m = Merger::new(1, t0, tx);
        // The board's count began at 0.5 s; records arrive 2-12 ms after
        // their last sample, and now and then one arrives after only 1 ms.
        let mut state = 3u32;
        let mut last_start = None;
        let mut biggest_step: f64 = 0.0;
        for n in 0..2000u64 {
            let t = 0.5 + 0.01 * n as f64; // a record every 10 ms
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let latency = if n % 97 == 0 { 0.001 } else { 0.002 + 0.01 * ((state >> 16) as f64 / 65536.0) };
            let last = ((t - 0.5) * BOARD_RATE) as u64;
            m.track(0, t0 + Duration::from_secs_f64(t + latency), last);
            let start = m.own_start(0, t);
            if t > 3.0 {
                if let Some(prev) = last_start {
                    biggest_step = biggest_step.max((start - prev as f64).abs());
                }
            }
            last_start = Some(start);
        }
        // Within the latency floor of the truth, and never a jump: between
        // records 10 ms apart it moves by no more than the fastest rate
        // allows over that time and the arrival jitter (a few us), where a
        // snapped estimate would move by milliseconds.
        let start = last_start.unwrap();
        assert!(start > 0.5 && start < 0.5 + 0.003, "start {}", start);
        assert!(biggest_step < 3e-6, "step {}", biggest_step);
    }

    #[test]
    fn timeline_follows_the_reference_count_not_host_time() {
        let (tx, _rx) = bounded::<Vec<i16>>(16);
        let t0 = Instant::now();
        let mut m = Merger::new(1, t0, tx);
        m.boards[0].start = Some(0.5);
        let first = m.position(0, 1_000_000.0, 1.0);
        // However the host-time estimate moves afterwards, bursts keep the
        // spacing of the board's own count.
        m.boards[0].start = Some(0.503);
        m.boards[0].rate = 80e-6;
        let second = m.position(0, 9_000_000.0, 9.0);
        assert_eq!(second - first, 8_000_000 * FACTOR as i64);
    }

    #[test]
    fn usb_frames_align_tiled_boards_and_track_drift() {
        let (tx, _rx) = bounded::<Vec<i16>>(16);
        let t0 = Instant::now();
        let mut m = Merger::new(2, t0, tx);
        m.reference = Some(0);
        let mut last = (0.0, 0.0);
        for n in 1..=40u64 {
            let reference = 500_000.0 + n as f64 * 4_096_000.0;
            let other = 1_000_000.0 + reference * (1.0 + 2e-6);
            let jitter = [-20, 8, 17, -5, 0][n as usize % 5] as f64;
            let frame = ((n * 256) & 0x7ff) as u16;
            let seen = t0 + Duration::from_secs_f64(n as f64 * 0.256);
            // Either board's record may reach the merger first. Frame-number
            // wrap is harmless because arrival time selects the same event.
            if n & 1 == 0 {
                m.usb_stamp(0, frame, reference.round() as u64, seen);
                m.usb_stamp(1, frame, (other + jitter).round() as u64, seen + Duration::from_millis(3));
            } else {
                m.usb_stamp(1, frame, (other + jitter).round() as u64, seen + Duration::from_millis(3));
                m.usb_stamp(0, frame, reference.round() as u64, seen);
            }
            last = (reference, other);
        }
        let mapped = m.reference_pair(1, last.1, 10.0);
        assert!((mapped - last.0).abs() < 5.0, "mapped {} instead of {}", mapped, last.0);
        let (_, drift, _) = m.boards[1].relative.unwrap();
        assert!((drift + 2e-6).abs() < 0.2e-6, "drift {}", drift);
        assert_eq!(m.boards[1].matches, 32);
    }

    #[test]
    fn only_tiled_boards_remove_the_dc_offset() {
        // A record mixed down by 1 MHz holding only the board's DC offset
        // (a tone at -1 MHz, a quarter turn back per sample).
        let word = |i: i32, q: i32| (i & 1023) as u32 | (((q & 1023) as u32) << 10);
        let pairs: Vec<u32> = (0..400)
            .map(|n| match n % 4 {
                0 => word(6, 0),
                1 => word(0, -6),
                2 => word(-6, 0),
                _ => word(0, 6),
            })
            .collect();
        let power = |s: &Shaped| {
            let x = s.channel.as_ref().unwrap();
            x.iter().map(|v| v.norm_sqr()).sum::<f32>() / x.len() as f32
        };
        // Folded boards can have a channel on the LO: left as received.
        let folded = Shaper::new(None).narrow(1000, 1, &pairs);
        assert!((power(&folded) - (6.0 * SAMPLE_SCALE as f32).powi(2)).abs() < 1.0);
        let tiled = Shaper::new(Some(tile_offset_half_mhz(0, 5, Layout::Tiled))).narrow(1000, 1, &pairs);
        assert!(power(&tiled) < 1e-3, "DC left: {}", power(&tiled));
    }

    #[test]
    fn tiled_boards_place_bursts_once_at_their_frequency() {
        // Board 4 of 5 tiled (LO 31.5 MHz above the centre) places the
        // centre of channelizer bin -7 only at +38.5 MHz. A real Bluetooth
        // channel at +39 retains its +0.5 MHz residual inside that bin.
        let shaper = Shaper::new(Some(tile_offset_half_mhz(4, 5, Layout::Tiled)));
        let s = shaper.narrow(1000, -7, &vec![100u32; 1000]);
        let mid = 2000..18000;
        assert!(share_at(&s.samples, mid.clone(), 38.5) > 0.99);
        assert!(share_at(&s.samples, mid, 22.5) < 1e-6);
        assert_eq!(
            (0..5)
                .map(|i| tile_offset_half_mhz(i, 5, Layout::Tiled))
                .collect::<Vec<_>>(),
            vec![-65, -33, -1, 31, 63]
        );
    }

    #[test]
    fn tiled_boards_cover_every_classic_channel() {
        let los: Vec<i64> = (0..5).map(|i| tile_offset_half_mhz(i, 5, Layout::Tiled)).collect();
        // Classic channels 0..78 are -39..+39 MHz around 2441. Each is no
        // farther than 7.5 MHz from exactly one LO, so the firmware's
        // channelizer (-7..+7 whole-MHz bins) retains it with a 0.5 MHz
        // residual instead of losing it on a tile boundary.
        for rf_mhz in -39..=39 {
            let rf_half_mhz = 2 * rf_mhz;
            let receivers = los.iter().filter(|&&lo| (rf_half_mhz - lo).abs() <= 15).count();
            assert_eq!(receivers, 1, "{} MHz has {} tiles", rf_mhz, receivers);
        }
    }

    #[test]
    fn ble_tiling_covers_ble_and_moves_advertising_channels_inside() {
        let los: Vec<i64> = (0..5).map(|i| tile_offset_half_mhz(i, 5, Layout::TiledBle)).collect();
        assert_eq!(los, [-65, -33, -1, 31, 65]);
        // BLE channels are 2402..2480 MHz in 2 MHz steps: -39..+39 MHz,
        // odd offsets from 2441.
        for rf_mhz in (-39..=39).step_by(2) {
            let receivers = los.iter().filter(|&&lo| (2 * rf_mhz - lo).abs() <= 15).count();
            assert_eq!(receivers, 1, "{} MHz has {} tiles", rf_mhz, receivers);
        }
        // Advertising channels 37, 38 and 39 (2402, 2426, 2480) sit at
        // least 1 MHz inside their tile.
        for rf_mhz in [-39i64, -15, 39] {
            assert!(los.iter().any(|&lo| (2 * rf_mhz - lo).abs() <= 13), "{} MHz is on a tile edge", rf_mhz);
        }
        // Relative +24 MHz is Classic channel 63 at 2465 MHz. This is the
        // one deliberate hole in this opt-in layout.
        assert!(!los.iter().any(|&lo| (48 - lo).abs() <= 15));
    }
}
