// Copyright 2026 CEMAXECUTER LLC

//! Receive clock for live capture.
//!
//! Packet timestamps are derived by counting samples from the moment
//! capture starts. Samples that never reach the host (an SDR overrun, a USB
//! hiccup, a receive thread blocked behind a slow decoder) would otherwise
//! vanish from that count, and every later timestamp would fall further
//! behind real time. `RxClock` compares the samples received with the
//! elapsed wall time and reports the shortfall as a gap, so the pipeline can
//! move its timestamps forward and count the time that was lost.

use std::time::Instant;

pub struct RxClock {
    rate: f64,
    start: Option<Instant>,
    /// Samples received since `start`, plus samples already reported lost.
    accounted: u64,
    /// Samples allowed to sit in driver buffers before a shortfall counts
    /// as lost.
    slack: u64,
    /// Smallest gap worth reporting.
    min_gap: u64,
    lost: u64,
}

impl RxClock {
    pub fn new(sample_rate: u32) -> Self {
        let rate = sample_rate as f64;
        Self {
            rate,
            start: None,
            accounted: 0,
            slack: (rate * 0.100) as u64,
            min_gap: (rate * 0.010) as u64,
            lost: 0,
        }
    }

    /// Record `n` samples that a receive call has just returned. Returns the
    /// number of samples lost before them, or 0 while the stream keeps up.
    ///
    /// The shortfall is measured against the whole run, so a gap reported a
    /// little early or late is corrected at the next one rather than adding
    /// up. Timestamps may trail real time by up to the slack, never lead it
    /// by more than what was buffered when a gap was reported.
    pub fn received(&mut self, n: usize, now: Instant) -> u64 {
        let start = match self.start {
            Some(s) => s,
            None => {
                // Samples in the first call were captured before it
                // returned; count time from here.
                self.start = Some(now);
                return 0;
            }
        };
        self.accounted += n as u64;
        let expected = (now.duration_since(start).as_secs_f64() * self.rate) as u64;
        let behind = expected.saturating_sub(self.accounted);
        if behind < self.slack + self.min_gap {
            return 0;
        }
        let gap = behind - self.slack;
        self.accounted += gap;
        self.lost += gap;
        gap
    }

    /// Samples reported lost so far.
    pub fn lost(&self) -> u64 {
        self.lost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const RATE: u32 = 1_000_000;

    fn feed(clock: &mut RxClock, start: Instant, ms: u64, chunk: usize) -> u64 {
        // Deliver `ms` milliseconds of samples in real time.
        let mut total = 0;
        let per_ms = RATE as usize / 1000;
        let mut sent = 0;
        for t in 1..=ms {
            sent += per_ms;
            if sent >= chunk {
                total += clock.received(sent, start + Duration::from_millis(t));
                sent = 0;
            }
        }
        total
    }

    #[test]
    fn continuous_stream_loses_nothing() {
        let mut c = RxClock::new(RATE);
        let t0 = Instant::now();
        c.received(1000, t0);
        assert_eq!(feed(&mut c, t0, 10_000, 4096), 0);
        assert_eq!(c.lost(), 0);
    }

    #[test]
    fn small_jitter_is_not_a_gap() {
        let mut c = RxClock::new(RATE);
        let t0 = Instant::now();
        c.received(1000, t0);
        // A receive call that returns 50 ms late, then catches up.
        assert_eq!(c.received(1000, t0 + Duration::from_millis(51)), 0);
        assert_eq!(c.received(50_000, t0 + Duration::from_millis(52)), 0);
        assert_eq!(c.lost(), 0);
    }

    #[test]
    fn stall_is_reported_once_and_not_double_counted() {
        let mut c = RxClock::new(RATE);
        let t0 = Instant::now();
        c.received(1000, t0);
        // 1 s of samples arrive on time.
        assert_eq!(c.received(1_000_000, t0 + Duration::from_secs(1)), 0);
        // Then the host stalls for 2 s and only 0.5 s of samples survive.
        let gap = c.received(500_000, t0 + Duration::from_secs(3));
        assert_eq!(gap, 1_500_000 - 100_000);
        // Streaming resumes on time: no further gap.
        assert_eq!(c.received(1_000_000, t0 + Duration::from_secs(4)), 0);
        assert_eq!(c.lost(), 1_400_000);
    }

    #[test]
    fn steady_loss_tracks_the_lost_fraction() {
        // Only 3 of every 4 seconds of samples reach the host.
        let mut c = RxClock::new(RATE);
        let t0 = Instant::now();
        c.received(1000, t0);
        let mut lost = 0;
        for s in 1..=400u64 {
            lost += c.received(750_000, t0 + Duration::from_secs(s));
        }
        let frac = lost as f64 / (400.0 * RATE as f64);
        assert!((frac - 0.25).abs() < 0.001, "lost fraction {frac}");
    }
}
