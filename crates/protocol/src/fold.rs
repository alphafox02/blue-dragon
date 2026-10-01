// Copyright 2025-2026 CEMAXECUTER LLC

//! Recognising one transmission decoded on several channels 16 MHz apart.
//!
//! A receiver that aliases (an ESP32-S3 at 16 Msps hears 80 MHz folded into
//! 16) gives the decoders the same burst at every channel it could have come
//! from. A transmitter sends on one channel at a time, so the same
//! transmitter decoded at the same instant on channels a multiple of 16 MHz
//! apart is one transmission seen through the fold, never two.

use std::collections::VecDeque;

use crate::Timespec;

/// How long decodes are remembered. Decodes from parallel workers arrive
/// out of order by a few batches, so this is well beyond that.
const MEMORY_NS: i64 = 50_000_000;
const MEMORY_ENTRIES: usize = 4096;

#[derive(Default)]
pub struct FoldMemory {
    /// Recent decodes: (transmitter key, time in ns, channel MHz).
    recent: VecDeque<(u32, i64, u32)>,
    /// Latest decode time seen.
    newest: i64,
}

impl FoldMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a decode from transmitter `key` (a LAP or access address) at
    /// `freq` MHz and time `ts`, and returns true if the same transmitter was
    /// already decoded within `window_ns` of it on a channel a multiple of
    /// 16 MHz away. A repeat is not recorded again.
    pub fn seen(&mut self, key: u32, freq: u32, ts: &Timespec, window_ns: i64) -> bool {
        let t = ns(ts);
        self.newest = self.newest.max(t);
        while let Some(&(_, old, _)) = self.recent.front() {
            if self.newest - old > MEMORY_NS {
                self.recent.pop_front();
            } else {
                break;
            }
        }
        let repeat = self.recent.iter().any(|&(k, at, f)| {
            let df = f.abs_diff(freq);
            k == key && (t - at).abs() <= window_ns && df != 0 && df % 16 == 0
        });
        if !repeat {
            if self.recent.len() >= MEMORY_ENTRIES {
                self.recent.pop_front();
            }
            self.recent.push_back((key, t, freq));
        }
        repeat
    }
}

/// Packets held back for a short while in case a better copy of the same
/// transmission (one that passes its CRC) is still to come from another
/// fold.
pub struct Held<T> {
    /// (transmitter key, time in ns, channel MHz, packet)
    held: VecDeque<(u32, i64, u32, T)>,
    newest: i64,
}

impl<T> Default for Held<T> {
    fn default() -> Self {
        Self { held: VecDeque::new(), newest: 0 }
    }
}

/// How long a packet is held, in packet time.
const HOLD_NS: i64 = 30_000_000;

impl<T> Held<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hold(&mut self, key: u32, freq: u32, ts: &Timespec, packet: T) {
        let t = ns(ts);
        self.newest = self.newest.max(t);
        self.held.push_back((key, t, freq, packet));
    }

    /// Drops held packets that are copies (within `window_ns`, a multiple of
    /// 16 MHz away) of a packet from `key` at `freq` and `ts`; returns how
    /// many.
    pub fn drop_copies_of(&mut self, key: u32, freq: u32, ts: &Timespec, window_ns: i64) -> usize {
        let t = ns(ts);
        self.newest = self.newest.max(t);
        let before = self.held.len();
        self.held.retain(|&(k, at, f, _)| {
            let df = f.abs_diff(freq);
            !(k == key && (t - at).abs() <= window_ns && df != 0 && df % 16 == 0)
        });
        before - self.held.len()
    }

    /// Takes out the packets held long enough, given packets seen up to `ts`.
    pub fn release(&mut self, ts: &Timespec) -> Vec<T> {
        self.newest = self.newest.max(ns(ts));
        let mut out = Vec::new();
        while self.held.front().is_some_and(|&(_, at, _, _)| self.newest - at > HOLD_NS) {
            if let Some((_, _, _, p)) = self.held.pop_front() {
                out.push(p);
            }
        }
        out
    }

    /// Takes out everything still held.
    pub fn release_all(&mut self) -> Vec<T> {
        self.held.drain(..).map(|(_, _, _, p)| p).collect()
    }
}

fn ns(ts: &Timespec) -> i64 {
    ts.tv_sec as i64 * 1_000_000_000 + ts.tv_nsec as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(ns: u64) -> Timespec {
        Timespec { tv_sec: 100, tv_nsec: ns }
    }

    #[test]
    fn folds_are_recognised() {
        let mut m = FoldMemory::new();
        let w = 10_000;
        assert!(!m.seen(0x123456, 2441, &at(1_000_000), w));
        // The same transmission folded 16 and 32 MHz away, in either order.
        assert!(m.seen(0x123456, 2457, &at(1_001_000), w));
        assert!(m.seen(0x123456, 2409, &at(999_500), w));
        // Not repeats: another transmitter, another spacing, the next slot.
        assert!(!m.seen(0x654321, 2457, &at(1_000_000), w));
        assert!(!m.seen(0x123456, 2450, &at(1_000_000), w));
        assert!(!m.seen(0x123456, 2441, &at(1_625_000), w));
        assert!(!m.seen(0x123456, 2457, &at(2_250_000), w));
        // A copy decoded after packets several milliseconds newer.
        assert!(!m.seen(0x123456, 2441, &at(9_000_000), w));
        assert!(m.seen(0x123456, 2473, &at(1_000_000), w));
    }

    #[test]
    fn old_decodes_are_forgotten() {
        let mut m = FoldMemory::new();
        assert!(!m.seen(1, 2441, &at(0), 10_000));
        assert!(!m.seen(2, 2441, &at(60_000_000), 10_000));
        assert_eq!(m.recent.len(), 1);
    }

    #[test]
    fn held_packets_wait_for_a_better_copy() {
        let mut h = Held::new();
        h.hold(7, 2464, &at(1_000_000), "ghost");
        h.hold(9, 2441, &at(1_000_000), "lone");
        // The valid copy 16 MHz away removes the ghost only.
        assert_eq!(h.drop_copies_of(7, 2480, &at(1_002_000), 20_000), 1);
        assert!(h.release(&at(20_000_000)).is_empty());
        assert_eq!(h.release(&at(40_000_000)), vec!["lone"]);
        h.hold(9, 2441, &at(50_000_000), "last");
        assert_eq!(h.release_all(), vec!["last"]);
    }
}
