use num_complex::Complex32;

/// AGC (Automatic Gain Control) replacing liquid-dsp's agc_crcf.
///
/// Uses a simple envelope follower with multiplicative gain update.
/// The gain is adapted to normalize the signal to unit amplitude.
/// Gain update uses the log-domain formula from liquid-dsp:
///   g *= exp(-0.5 * alpha * ln(y_hat))
/// Always positive, never oscillates (unlike the linear approximation
/// which goes negative for signal_level > 1/alpha + 1).
pub struct Agc {
    gain: f32,
    bandwidth: f32,
    signal_level: f32,
    /// Squelch threshold in dB
    squelch_threshold: f32,
    /// Squelch timeout in samples
    squelch_timeout: u32,
    /// Current squelch state
    state: SquelchState,
    /// Counter for timeout
    timer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SquelchState {
    /// Signal is below threshold (quiet)
    SignalLo,
    /// Signal just crossed above threshold (rising edge)
    Rise,
    /// Signal is above threshold (active)
    SignalHi,
    /// Signal just went below threshold and timed out (falling edge)
    Timeout,
}

impl Agc {
    pub fn new(bandwidth: f32, squelch_threshold_db: f32, squelch_timeout: u32) -> Self {
        Self {
            gain: 1000.0, // start with high gain (low signal assumption)
            bandwidth,
            signal_level: 1e-3,
            squelch_threshold: squelch_threshold_db,
            squelch_timeout,
            state: SquelchState::SignalLo,
            timer: 0,
        }
    }

    /// Get current RSSI estimate in dB
    pub fn rssi_db(&self) -> f32 {
        -20.0 * self.gain.log10()
    }

    /// Process one sample through AGC
    pub fn execute(&mut self, input: Complex32) -> (Complex32, SquelchState) {
        // Apply gain
        let output = input * self.gain;

        // Update signal level estimate (envelope follower)
        let level = output.norm();
        let alpha = self.bandwidth;
        self.signal_level = alpha * level + (1.0 - alpha) * self.signal_level;

        // Adapt gain toward unity output using liquid-dsp's stable formula:
        // g *= exp(-0.5 * alpha * ln(y_hat))
        // Smoothly decreases gain when output > 1, increases when < 1.
        if self.signal_level > 1e-6 {
            self.gain *= (-0.5 * alpha * self.signal_level.ln()).exp();
            self.gain = self.gain.clamp(1e-6, 1e6);
        }

        // Squelch state machine
        let rssi = self.rssi_db();
        let above_threshold = rssi > self.squelch_threshold;

        let prev_state = self.state;
        self.state = match prev_state {
            SquelchState::SignalLo => {
                if above_threshold {
                    SquelchState::Rise
                } else {
                    SquelchState::SignalLo
                }
            }
            SquelchState::Rise => {
                if above_threshold {
                    self.timer = 0;
                    SquelchState::SignalHi
                } else {
                    self.timer = 0;
                    SquelchState::SignalLo
                }
            }
            SquelchState::SignalHi => {
                if above_threshold {
                    self.timer = 0;
                    SquelchState::SignalHi
                } else {
                    self.timer += 1;
                    if self.timer >= self.squelch_timeout {
                        SquelchState::Timeout
                    } else {
                        SquelchState::SignalHi
                    }
                }
            }
            SquelchState::Timeout => {
                if above_threshold {
                    SquelchState::Rise
                } else {
                    SquelchState::SignalLo
                }
            }
        };

        (output, self.state)
    }

    /// Set squelch threshold
    pub fn set_squelch_threshold(&mut self, threshold_db: f32) {
        self.squelch_threshold = threshold_db;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agc_gain_convergence() {
        let mut agc = Agc::new(0.25, -45.0, 100);

        // Feed a constant-amplitude signal
        let signal_level = 0.01f32;
        for _ in 0..1000 {
            let sample = Complex32::new(signal_level, 0.0);
            let (output, _) = agc.execute(sample);
            let _ = output;
        }

        // After convergence, output should be near unity
        let sample = Complex32::new(signal_level, 0.0);
        let (output, _) = agc.execute(sample);
        assert!(
            (output.norm() - 1.0).abs() < 0.3,
            "AGC output norm = {}, expected ~1.0",
            output.norm()
        );
    }

    #[test]
    fn test_agc_stability_strong_signal() {
        let mut agc = Agc::new(0.25, -45.0, 100);

        // Start with weak noise
        for _ in 0..100 {
            agc.execute(Complex32::new(0.001, 0.0));
        }

        // Sudden strong signal (50x noise)
        for _ in 0..50 {
            let (output, _) = agc.execute(Complex32::new(0.05, 0.0));
            assert!(output.norm() > 0.0, "output should never be zero");
        }

        // After convergence, should be near unity
        let (output, _) = agc.execute(Complex32::new(0.05, 0.0));
        assert!(
            (output.norm() - 1.0).abs() < 0.5,
            "AGC output = {}, expected ~1.0",
            output.norm()
        );
    }

    #[test]
    fn test_squelch_state_machine() {
        let mut agc = Agc::new(0.25, -60.0, 5);

        // Feed silence -> should stay SignalLo
        for _ in 0..50 {
            let (_, state) = agc.execute(Complex32::new(0.0, 0.0));
            assert!(
                state == SquelchState::SignalLo,
                "expected SignalLo during silence, got {:?}",
                state
            );
        }
    }
}
