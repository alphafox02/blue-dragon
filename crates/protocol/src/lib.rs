pub mod ble;
pub mod ble_connection;
pub mod btbb;
pub mod fec;

/// Frequency in MHz to BLE channel number mapping
pub fn freq_to_channel(freq_mhz: u32) -> u32 {
    let phys_channel = (freq_mhz - 2402) / 2;
    match phys_channel {
        0 => 37,
        12 => 38,
        39 => 39,
        c if c < 12 => c - 1,
        c => c - 2,
    }
}

/// Common packet types shared across protocol layers
#[derive(Debug, Clone)]
pub struct Timespec {
    pub tv_sec: u64,
    pub tv_nsec: u64,
}

impl Default for Timespec {
    fn default() -> Self {
        Self { tv_sec: 0, tv_nsec: 0 }
    }
}
