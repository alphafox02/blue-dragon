// Copyright 2026 CEMAXECUTER LLC
//
// BLE MITM relay: position blue-dragon between two BLE devices by
// impersonating each side and relaying traffic.
//
// Architecture:
//   Victim Central  <--HCI adapter (peripheral mode)-->  Blue Dragon  <--WHAD/HCI2 (central mode)-->  Target Peripheral
//
// Phase 1: Clone target's advertisement via HCI adapter
// Phase 2: Accept connection from victim central on HCI
// Phase 3: Connect to real target via WHAD or second HCI
// Phase 4: Relay L2CAP/ATT traffic bidirectionally
// Phase 5: Optionally intercept/modify/inject packets in transit
//
// Requires: one HCI adapter (for peripheral side) + WHAD dongle or
// second HCI adapter (for central side).

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// MITM relay configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitmConfig {
    /// Target device MAC address (the peripheral to impersonate)
    pub target_mac: String,
    /// Clone target's advertisement data
    pub clone_adv: bool,
    /// Custom name to advertise (overrides cloned name if set)
    pub custom_name: Option<String>,
    /// Whether to log all relayed packets
    pub log_traffic: bool,
    /// Optional packet modification rules (future)
    pub modify_rules: Vec<String>,
}

/// State of the MITM relay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MitmState {
    /// Not started
    Idle,
    /// Scanning for target device
    Scanning,
    /// Cloning target's advertisement
    Cloning,
    /// Waiting for victim to connect
    WaitingForVictim,
    /// Connecting to real target
    ConnectingToTarget,
    /// Relaying traffic
    Relaying,
    /// Stopped
    Stopped,
}

/// MITM relay event for logging/monitoring.
#[derive(Debug, Clone)]
pub enum MitmEvent {
    StateChange(MitmState),
    TargetFound { mac: String, name: Option<String> },
    VictimConnected { mac: String },
    TargetConnected,
    PacketRelayed { direction: String, len: usize },
    PacketModified { direction: String, original_len: usize, modified_len: usize },
    Error(String),
    Stopped,
}

/// MITM relay controller.
/// Coordinates between HCI adapter (peripheral side) and WHAD/HCI2 (central side).
pub struct MitmRelay {
    config: MitmConfig,
    state: MitmState,
    running: Arc<AtomicBool>,
    events: Vec<MitmEvent>,
}

impl MitmRelay {
    pub fn new(config: MitmConfig) -> Self {
        Self {
            config,
            state: MitmState::Idle,
            running: Arc::new(AtomicBool::new(false)),
            events: Vec::new(),
        }
    }

    /// Get current relay state.
    pub fn state(&self) -> MitmState {
        self.state
    }

    /// Get pending events (drains the queue).
    pub fn drain_events(&mut self) -> Vec<MitmEvent> {
        std::mem::take(&mut self.events)
    }

    /// Start the MITM relay. This is a blocking call that runs the relay
    /// until stopped or an error occurs.
    ///
    /// `hci_adapter`: BlueZ adapter name (e.g., "hci0") for peripheral side
    /// `whad_port`: WHAD serial port (e.g., "/dev/ttyACM0") for central side
    ///              If None, uses a second HCI adapter
    pub fn start(
        &mut self,
        _hci_adapter: &str,
        _whad_port: Option<&str>,
    ) -> Result<(), String> {
        self.running.store(true, Ordering::SeqCst);
        self.set_state(MitmState::Scanning);

        // Phase 1: Scan for target device and capture its advertisement data
        eprintln!("MITM: scanning for target {}", self.config.target_mac);
        // TODO: Use HCI active scan to find target and capture adv data
        // For now, this is a framework that will be connected to the
        // actual HCI/WHAD backends during integration testing.

        self.set_state(MitmState::Cloning);
        eprintln!("MITM: cloning target advertisement");
        // TODO: Start advertising with cloned data via spoof_advertisement()

        self.set_state(MitmState::WaitingForVictim);
        eprintln!("MITM: waiting for victim to connect");
        // TODO: Accept incoming connection via BlueZ peripheral mode

        self.set_state(MitmState::ConnectingToTarget);
        eprintln!("MITM: connecting to real target via WHAD");
        // TODO: Connect to target via WHAD or second HCI

        self.set_state(MitmState::Relaying);
        eprintln!("MITM: relaying traffic");
        // TODO: Bidirectional L2CAP relay loop

        // Placeholder: wait until stopped
        while self.running.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        self.set_state(MitmState::Stopped);
        self.events.push(MitmEvent::Stopped);
        Ok(())
    }

    /// Stop the relay.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Get a clone of the running flag for external stop control.
    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    fn set_state(&mut self, state: MitmState) {
        self.state = state;
        self.events.push(MitmEvent::StateChange(state));
    }
}
