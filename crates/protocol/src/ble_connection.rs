// Copyright 2025-2026 CEMAXECUTER LLC

use crate::ble::{BlePacket, BLE_ADV_AA};

pub const BLE_MAX_CONNECTIONS: usize = 128;

/// BLE connection tracking (populated from CONNECT_IND packets)
#[derive(Debug, Clone)]
pub struct BleConnection {
    pub aa: u32,
    pub crc_init: u32,
    pub init_addr: [u8; 6],
    pub adv_addr: [u8; 6],
    pub init_addr_type: u8,
    pub adv_addr_type: u8,
    pub channel_map: [u8; 5],
    pub hop_increment: u8,
    pub interval: u16,
    pub latency: u16,
    pub timeout: u16,
    pub created: u64,
    pub last_seen: u64,
    pub pkt_count: u32,
    pub active: bool,
    /// Precomputed list of used data channels from channel_map
    pub used_channels: Vec<u8>,
    /// Last unmapped channel for CSA#1 hop prediction
    pub last_unmapped_channel: u8,
    /// Connection event counter (incremented each interval)
    pub event_counter: u16,
    /// Whether we've anchored to the hop sequence
    pub hop_anchored: bool,
    /// Predicted next data channel
    pub predicted_channel: Option<u8>,
}

impl Default for BleConnection {
    fn default() -> Self {
        Self {
            aa: 0,
            crc_init: 0,
            init_addr: [0; 6],
            adv_addr: [0; 6],
            init_addr_type: 0,
            adv_addr_type: 0,
            channel_map: [0; 5],
            hop_increment: 0,
            interval: 0,
            latency: 0,
            timeout: 0,
            created: 0,
            last_seen: 0,
            pkt_count: 0,
            active: false,
            used_channels: Vec::new(),
            last_unmapped_channel: 0,
            event_counter: 0,
            hop_anchored: false,
            predicted_channel: None,
        }
    }
}

/// Compute the list of used data channels from a 5-byte channel map bitmask.
/// BLE data channels 0-36 are mapped: bit i of byte[i/8] indicates channel i.
pub fn channels_from_map(ch_map: &[u8; 5]) -> Vec<u8> {
    let mut used = Vec::with_capacity(37);
    for ch in 0u8..37 {
        let byte_idx = (ch / 8) as usize;
        let bit_idx = ch % 8;
        if ch_map[byte_idx] & (1 << bit_idx) != 0 {
            used.push(ch);
        }
    }
    used
}

/// BLE Channel Selection Algorithm #1 (CSA#1).
/// BT Core Spec Vol 6 Part B Section 4.5.8.
///
/// Given the last unmapped channel and hop increment, compute the next
/// data channel. Returns (data_channel, unmapped_channel).
pub fn csa1_next_channel(
    last_unmapped: u8,
    hop_increment: u8,
    used_channels: &[u8],
) -> (u8, u8) {
    let unmapped = (last_unmapped + hop_increment) % 37;

    // Check if unmapped channel is in the used set
    if used_channels.contains(&unmapped) {
        (unmapped, unmapped)
    } else {
        // Remap: index into used channel list
        let remap_idx = (unmapped as usize) % used_channels.len();
        (used_channels[remap_idx], unmapped)
    }
}

/// Predict the next N data channels for a connection.
pub fn predict_channels(
    conn: &BleConnection,
    count: usize,
) -> Vec<u8> {
    if conn.used_channels.is_empty() || conn.hop_increment == 0 {
        return Vec::new();
    }

    let mut predictions = Vec::with_capacity(count);
    let mut last_unmapped = conn.last_unmapped_channel;

    for _ in 0..count {
        let (data_ch, unmapped) = csa1_next_channel(
            last_unmapped,
            conn.hop_increment,
            &conn.used_channels,
        );
        predictions.push(data_ch);
        last_unmapped = unmapped;
    }

    predictions
}

/// Connection table: tracks active BLE connections from CONNECT_IND PDUs
pub struct ConnectionTable {
    slots: Vec<BleConnection>,
}

impl ConnectionTable {
    pub fn new() -> Self {
        let mut slots = Vec::with_capacity(BLE_MAX_CONNECTIONS);
        for _ in 0..BLE_MAX_CONNECTIONS {
            slots.push(BleConnection::default());
        }
        Self { slots }
    }

    /// Count active connections
    pub fn count(&self) -> usize {
        self.slots.iter().filter(|c| c.active).count()
    }

    /// Get all active connections
    pub fn active_connections(&self) -> Vec<&BleConnection> {
        self.slots.iter().filter(|c| c.active).collect()
    }

    /// Look up a connection by access address. Returns mutable reference.
    pub fn lookup_mut(&mut self, aa: u32) -> Option<&mut BleConnection> {
        self.slots.iter_mut().find(|c| c.active && c.aa == aa)
    }

    /// Look up a connection by access address. Returns shared reference.
    pub fn lookup(&self, aa: u32) -> Option<&BleConnection> {
        self.slots.iter().find(|c| c.active && c.aa == aa)
    }

    /// Add or update a connection from parsed CONNECT_IND fields.
    /// Returns true if a new connection was added.
    pub fn add(
        &mut self,
        aa: u32,
        crc_init: u32,
        init_addr: &[u8; 6],
        adv_addr: &[u8; 6],
        init_type: u8,
        adv_type: u8,
        ch_map: &[u8; 5],
        hop: u8,
        interval: u16,
        latency: u16,
        timeout: u16,
        now: u64,
    ) -> bool {
        // Check if AA already tracked
        if let Some(c) = self.lookup_mut(aa) {
            c.last_seen = now;
            return false;
        }

        // Find free slot or evict oldest
        let mut best = 0;
        let mut oldest: u64 = u64::MAX;
        for i in 0..BLE_MAX_CONNECTIONS {
            if !self.slots[i].active {
                best = i;
                break;
            }
            if self.slots[i].last_seen < oldest {
                oldest = self.slots[i].last_seen;
                best = i;
            }
        }

        let c = &mut self.slots[best];
        *c = BleConnection::default();
        c.active = true;
        c.aa = aa;
        c.crc_init = crc_init;
        c.init_addr.copy_from_slice(init_addr);
        c.adv_addr.copy_from_slice(adv_addr);
        c.init_addr_type = init_type;
        c.adv_addr_type = adv_type;
        c.channel_map.copy_from_slice(ch_map);
        c.hop_increment = hop;
        c.interval = interval;
        c.latency = latency;
        c.timeout = timeout;
        c.created = now;
        c.last_seen = now;
        c.used_channels = channels_from_map(ch_map);

        true
    }

    /// Parse a CONNECT_IND advertising PDU and register the connection.
    /// p.data layout: AA(4) + Header(2) + InitA(6) + AdvA(6) + LLData(22) + CRC(3) = 43 bytes
    /// Returns the connection AA if successfully parsed, None otherwise.
    pub fn parse_connect_ind(&mut self, p: &BlePacket, now: u64) -> Option<u32> {
        if p.aa != BLE_ADV_AA {
            return None;
        }
        if p.len < 43 {
            return None;
        }

        let pdu_type = p.data[4] & 0x0F;
        if pdu_type != 5 {
            return None;
        }

        let pdu_len = p.data[5];
        if pdu_len != 34 {
            return None;
        }

        let init_type = (p.data[4] >> 6) & 1; // TxAdd
        let adv_type = (p.data[4] >> 7) & 1; // RxAdd

        let init_addr: [u8; 6] = p.data[6..12].try_into().ok()?;
        let adv_addr: [u8; 6] = p.data[12..18].try_into().ok()?;

        // LLData starts at byte 18
        let conn_aa = u32::from_le_bytes(p.data[18..22].try_into().ok()?);
        let crc_init = p.data[22] as u32
            | ((p.data[23] as u32) << 8)
            | ((p.data[24] as u32) << 16);
        // byte 25: WinSize (skip)
        // bytes 26-27: WinOffset (skip)
        let interval = u16::from_le_bytes(p.data[28..30].try_into().ok()?);
        let latency = u16::from_le_bytes(p.data[30..32].try_into().ok()?);
        let timeout = u16::from_le_bytes(p.data[32..34].try_into().ok()?);
        let ch_map: [u8; 5] = p.data[34..39].try_into().ok()?;
        let hop = p.data[39] & 0x1F;

        self.add(
            conn_aa, crc_init, &init_addr, &adv_addr,
            init_type, adv_type, &ch_map, hop,
            interval, latency, timeout, now,
        );

        Some(conn_aa)
    }

    /// Record an observed data channel for a connection and try to anchor
    /// the hop sequence. If already anchored, advance the prediction.
    /// Returns the next predicted channel if anchored.
    pub fn observe_data_channel(&mut self, aa: u32, observed_channel: u8) -> Option<u8> {
        let conn = self.lookup_mut(aa)?;
        if conn.used_channels.is_empty() {
            return None;
        }

        if !conn.hop_anchored {
            // Try to anchor: find which unmapped value produces this data channel
            // by trying all 37 possible last_unmapped values
            for candidate in 0u8..37 {
                let (data_ch, _) = csa1_next_channel(candidate, conn.hop_increment, &conn.used_channels);
                if data_ch == observed_channel {
                    let unmapped = (candidate + conn.hop_increment) % 37;
                    conn.last_unmapped_channel = unmapped;
                    conn.hop_anchored = true;
                    // Predict next
                    let (next_ch, next_unmapped) = csa1_next_channel(
                        conn.last_unmapped_channel,
                        conn.hop_increment,
                        &conn.used_channels,
                    );
                    conn.last_unmapped_channel = next_unmapped;
                    conn.predicted_channel = Some(next_ch);
                    conn.event_counter = conn.event_counter.wrapping_add(1);
                    return Some(next_ch);
                }
            }
            None
        } else {
            // Already anchored: check if prediction matches
            let matched = conn.predicted_channel == Some(observed_channel);
            if !matched {
                // Lost sync -- re-anchor
                conn.hop_anchored = false;
                conn.predicted_channel = None;
                return self.observe_data_channel(aa, observed_channel);
            }

            // Advance to next hop
            let (next_ch, next_unmapped) = csa1_next_channel(
                conn.last_unmapped_channel,
                conn.hop_increment,
                &conn.used_channels,
            );
            conn.last_unmapped_channel = next_unmapped;
            conn.predicted_channel = Some(next_ch);
            conn.event_counter = conn.event_counter.wrapping_add(1);
            Some(next_ch)
        }
    }

    /// CRC init lookup function suitable for passing to ble_burst().
    /// Returns Some((crc_init, true)) if connection found, None if unknown.
    /// Also updates last_seen and pkt_count for tracked connections.
    pub fn crc_init_for_aa(&mut self, aa: u32, now: u64) -> Option<(u32, bool)> {
        if let Some(c) = self.lookup_mut(aa) {
            c.last_seen = now;
            c.pkt_count += 1;
            Some((c.crc_init, true))
        } else {
            None
        }
    }
}

impl Default for ConnectionTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_table_add_lookup() {
        let mut table = ConnectionTable::new();
        let init_addr = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let adv_addr = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let ch_map = [0xFF, 0xFF, 0xFF, 0xFF, 0x1F];

        assert_eq!(table.count(), 0);

        let added = table.add(
            0x12345678, 0xABCDEF, &init_addr, &adv_addr,
            0, 1, &ch_map, 7, 24, 0, 200, 1000,
        );
        assert!(added);
        assert_eq!(table.count(), 1);

        let conn = table.lookup(0x12345678).unwrap();
        assert_eq!(conn.crc_init, 0xABCDEF);
        assert_eq!(conn.hop_increment, 7);
        assert_eq!(conn.interval, 24);

        // Adding same AA should not create new entry
        let added = table.add(
            0x12345678, 0xABCDEF, &init_addr, &adv_addr,
            0, 1, &ch_map, 7, 24, 0, 200, 2000,
        );
        assert!(!added);
        assert_eq!(table.count(), 1);
    }

    #[test]
    fn test_channels_from_map() {
        // All 37 channels used
        let full_map = [0xFF, 0xFF, 0xFF, 0xFF, 0x1F];
        let used = channels_from_map(&full_map);
        assert_eq!(used.len(), 37);
        assert_eq!(used[0], 0);
        assert_eq!(used[36], 36);
    }

    #[test]
    fn test_channels_from_map_sparse() {
        // Only channels 0, 1, 2 used (bits 0-2 of byte 0)
        let map = [0x07, 0x00, 0x00, 0x00, 0x00];
        let used = channels_from_map(&map);
        assert_eq!(used, vec![0, 1, 2]);
    }

    #[test]
    fn test_csa1_next_channel_used() {
        // All channels used, hop=7, last_unmapped=0
        let used: Vec<u8> = (0..37).collect();
        let (data_ch, unmapped) = csa1_next_channel(0, 7, &used);
        assert_eq!(unmapped, 7);
        assert_eq!(data_ch, 7); // channel 7 is in the used set
    }

    #[test]
    fn test_csa1_next_channel_remapped() {
        // Only channels 0, 10, 20 used. hop=5, last_unmapped=0
        // unmapped = (0 + 5) % 37 = 5. Channel 5 not in used set.
        // remap_idx = 5 % 3 = 2. used[2] = 20.
        let used = vec![0, 10, 20];
        let (data_ch, unmapped) = csa1_next_channel(0, 5, &used);
        assert_eq!(unmapped, 5);
        assert_eq!(data_ch, 20);
    }

    #[test]
    fn test_predict_channels() {
        let mut conn = BleConnection::default();
        conn.hop_increment = 7;
        conn.channel_map = [0xFF, 0xFF, 0xFF, 0xFF, 0x1F]; // all 37
        conn.used_channels = channels_from_map(&conn.channel_map);
        conn.last_unmapped_channel = 0;

        let predictions = predict_channels(&conn, 5);
        assert_eq!(predictions.len(), 5);
        // With all 37 channels: 7, 14, 21, 28, 35
        assert_eq!(predictions, vec![7, 14, 21, 28, 35]);
    }

    #[test]
    fn test_predict_channels_wraparound() {
        let mut conn = BleConnection::default();
        conn.hop_increment = 13;
        conn.channel_map = [0xFF, 0xFF, 0xFF, 0xFF, 0x1F]; // all 37
        conn.used_channels = channels_from_map(&conn.channel_map);
        conn.last_unmapped_channel = 30;

        let predictions = predict_channels(&conn, 4);
        // 30+13=43%37=6, 6+13=19, 19+13=32, 32+13=45%37=8
        assert_eq!(predictions, vec![6, 19, 32, 8]);
    }

    #[test]
    fn test_connection_table_eviction() {
        let mut table = ConnectionTable::new();
        let init_addr = [0; 6];
        let adv_addr = [0; 6];
        let ch_map = [0xFF; 5];

        // Fill all slots
        for i in 0..BLE_MAX_CONNECTIONS {
            table.add(
                i as u32 + 1, 0x111111, &init_addr, &adv_addr,
                0, 0, &ch_map, 5, 24, 0, 200, i as u64,
            );
        }
        assert_eq!(table.count(), BLE_MAX_CONNECTIONS);

        // Adding one more should evict the oldest (last_seen = 0)
        table.add(
            0xFFFFFFFF, 0x222222, &init_addr, &adv_addr,
            0, 0, &ch_map, 5, 24, 0, 200, 1000,
        );
        assert_eq!(table.count(), BLE_MAX_CONNECTIONS);

        // The new connection should be findable
        assert!(table.lookup(0xFFFFFFFF).is_some());
    }
}
