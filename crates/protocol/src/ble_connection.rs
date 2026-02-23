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
        }
    }
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
