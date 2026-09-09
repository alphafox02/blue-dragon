// Copyright 2025-2026 CEMAXECUTER LLC

use bd_protocol::ble::{BlePacket, BlePhy};
use bd_protocol::btbb::ClassicBtPacket;
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{self, Write};

// Link-layer type constants
pub const DLT_PPI: u32 = 192;
pub const DLT_BLUETOOTH_LE_LL_WITH_PHDR: u32 = 256;
pub const DLT_BLUETOOTH_BREDR_BB: u32 = 255;

// BLE LE header flags
const LE_DEWHITENED: u16 = 0x0001;
const LE_SIGNAL_POWER_VALID: u16 = 0x0002;
const LE_NOISE_POWER_VALID: u16 = 0x0004;
const LE_CRC_CHECKED: u16 = 0x0400;
const LE_CRC_VALID: u16 = 0x0800;

// PHY encoding in flags bits 14-15 (per LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR spec)
const LE_PHY_1M: u16 = 0x0000;
const LE_PHY_2M: u16 = 0x4000;
const LE_PHY_CODED: u16 = 0x8000;

// Classic BT BR/EDR header flags
const BREDR_DEWHITENED: u16 = 0x0001;
const BREDR_SIGNAL_POWER_VALID: u16 = 0x0002;
const BREDR_NOISE_POWER_VALID: u16 = 0x0004;
// UAP/header validity bits (values per libbtbb pcap-common.h so Wireshark reads them)
const BREDR_REFLAP_VALID: u16 = 0x0010;
const BREDR_REFUAP_VALID: u16 = 0x0080;
const BREDR_HEC_CHECKED: u16 = 0x0100;
const BREDR_HEC_VALID: u16 = 0x0200;
const BREDR_PAYLOAD_PRESENT: u16 = 0x0020;
const BREDR_CRC_CHECKED: u16 = 0x0400;
const BREDR_CRC_VALID: u16 = 0x0800;

/// Payload flag bits for a decoded BR payload (present / CRC checked / CRC valid).
fn bredr_payload_flags(pkt: &ClassicBtPacket) -> u16 {
    if pkt.decoded_payload.is_empty() {
        0
    } else {
        BREDR_DEWHITENED
            | BREDR_PAYLOAD_PRESENT
            | BREDR_CRC_CHECKED
            | if pkt.crc_ok { BREDR_CRC_VALID } else { 0 }
    }
}

fn bredr_transport_rate(pkt: &ClassicBtPacket) -> u8 {
    if pkt.decoded_payload.is_empty() {
        return 0;
    }
    match pkt
        .header
        .and_then(|header| bd_protocol::btbb::edr_bits_per_symbol(header.pkt_type))
    {
        Some(2) => 0x31, // ACL, pi/4-DQPSK
        Some(3) => 0x32, // ACL, 8DPSK
        _ => 0x30,       // ACL, BR/GFSK
    }
}

/// Pack the recovered address into the BR/EDR `ref_lap_uap` field and the decoded
/// 18-bit `bt_header`, and OR in the matching validity flags.
fn bredr_addr_fields(pkt: &ClassicBtPacket) -> (u32, u32, u16) {
    let mut flags = BREDR_REFLAP_VALID;
    let ref_lap_uap = match (pkt.uap, pkt.uap_verified) {
        (Some(uap), true) => {
            flags |= BREDR_REFUAP_VALID;
            ((uap as u32) << 24) | (pkt.lap & 0xff_ffff)
        }
        _ => pkt.lap & 0xff_ffff,
    };
    let bt_header = match pkt.header {
        Some(h) => {
            if pkt.uap_verified {
                flags |= BREDR_HEC_CHECKED | BREDR_HEC_VALID;
            }
            (h.lt_addr as u32 & 0x7)
                | ((h.pkt_type as u32 & 0xf) << 3)
                | ((h.flow as u32 & 1) << 7)
                | ((h.arqn as u32 & 1) << 8)
                | ((h.seqn as u32 & 1) << 9)
                | ((h.hec as u32) << 10)
        }
        None => 0,
    };
    (ref_lap_uap, bt_header, flags)
}

/// Map BLE PHY type to PCAP flags bits 14-15
fn phy_to_flags(phy: BlePhy) -> u16 {
    match phy {
        BlePhy::Phy1M => LE_PHY_1M,
        BlePhy::Phy2M => LE_PHY_2M,
        BlePhy::PhyCoded => LE_PHY_CODED,
    }
}

// ZMQ packet type prefix bytes
pub const ZMQ_PKT_TYPE_BLE: u8 = 0x00;
pub const ZMQ_PKT_TYPE_BT: u8 = 0x01;

// PPI GPS field type
pub const PPI_FIELD_GPS: u16 = 30002;

// PPI header size (no fields): version(1) + flags(1) + len(2) + dlt(4) = 8
const PPI_HDR_SIZE: usize = 8;
// PPI GPS field: field_hdr(4) + gps_data(24) = 28
const PPI_FIELD_HDR_SIZE: usize = 4;
const PPI_GPS_DATA_SIZE: usize = 24;
const PPI_GPS_SIZE: usize = PPI_HDR_SIZE + PPI_FIELD_HDR_SIZE + PPI_GPS_DATA_SIZE;

// PPI-GPS present flags
const PPI_GPS_FLAG_GPSFLAGS: u32 = 0x00000001;
const PPI_GPS_FLAG_LAT: u32 = 0x00000002;
const PPI_GPS_FLAG_LON: u32 = 0x00000004;
const PPI_GPS_FLAG_ALT: u32 = 0x00000008;

/// GPS fix data
#[derive(Debug, Clone, Default)]
pub struct GpsFix {
    pub valid: bool,
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
}

/// ZMQ GPS frame (24 bytes: 3x f64 LE)
#[derive(Debug, Clone, Copy)]
pub struct ZmqGpsFrame {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
}

/// Convert double to PPI fixed3_7 format (for lat/lon, unsigned with +180 offset).
/// Clamps to valid range to prevent garbage from corrupted GPS data.
fn ppi_fixed3_7(val: f64) -> u32 {
    let clamped = val.clamp(-180.0, 180.0);
    ((clamped + 180.0) * 1e7) as u32
}

/// Convert double to PPI fixed6_4 format (for altitude, unsigned with +180000m offset).
/// Clamps to prevent underflow from invalid altitude values.
fn ppi_fixed6_4(val: f64) -> u32 {
    let clamped = val.clamp(-180000.0, 900000.0);
    ((clamped + 180000.0) * 1e4) as u32
}

/// Write PPI header
fn write_ppi_header<W: Write>(w: &mut W, ppi_len: u16, dlt: u32) -> io::Result<()> {
    w.write_u8(0)?; // version
    w.write_u8(0)?; // flags
    w.write_u16::<LittleEndian>(ppi_len)?;
    w.write_u32::<LittleEndian>(dlt)?;
    Ok(())
}

/// Write PPI GPS field
fn write_ppi_gps<W: Write>(w: &mut W, fix: &GpsFix) -> io::Result<()> {
    // Field header
    w.write_u16::<LittleEndian>(PPI_FIELD_GPS)?; // type
    w.write_u16::<LittleEndian>(PPI_GPS_DATA_SIZE as u16)?; // datalen

    // GPS data
    w.write_u8(2)?; // geotag_ver
    w.write_u8(0)?; // pad
    w.write_u16::<LittleEndian>(PPI_GPS_DATA_SIZE as u16)?; // geotag_len
    w.write_u32::<LittleEndian>(
        PPI_GPS_FLAG_GPSFLAGS | PPI_GPS_FLAG_LAT | PPI_GPS_FLAG_LON | PPI_GPS_FLAG_ALT,
    )?; // present_flags
    w.write_u32::<LittleEndian>(0)?; // gps_flags
    w.write_u32::<LittleEndian>(ppi_fixed3_7(fix.latitude))?;
    w.write_u32::<LittleEndian>(ppi_fixed3_7(fix.longitude))?;
    w.write_u32::<LittleEndian>(ppi_fixed6_4(fix.altitude))?;
    Ok(())
}

/// PCAP file writer
pub struct PcapWriter<W: Write> {
    writer: W,
}

impl<W: Write> PcapWriter<W> {
    /// Create a new PCAP file writer with the global header.
    /// Always uses DLT_PPI to support mixed BLE + Classic BT per-packet DLT.
    pub fn new(mut writer: W) -> io::Result<Self> {
        // PCAP global header
        writer.write_u32::<LittleEndian>(0xa1b2c3d4)?; // magic
        writer.write_u16::<LittleEndian>(2)?; // version_major
        writer.write_u16::<LittleEndian>(4)?; // version_minor
        writer.write_i32::<LittleEndian>(0)?; // thiszone
        writer.write_u32::<LittleEndian>(0)?; // sigfigs
        let snaplen = PPI_GPS_SIZE as u32 + 4 + 2 + 255 + 3;
        writer.write_u32::<LittleEndian>(snaplen)?; // snaplen
        writer.write_u32::<LittleEndian>(DLT_PPI)?; // network
        writer.flush()?;
        Ok(Self { writer })
    }

    /// Write a BLE packet record
    pub fn write_ble(&mut self, pkt: &BlePacket, gps: Option<&GpsFix>) -> io::Result<()> {
        let mut flags: u16 = LE_DEWHITENED | LE_SIGNAL_POWER_VALID | LE_NOISE_POWER_VALID;
        if pkt.crc_checked {
            flags |= LE_CRC_CHECKED;
            if pkt.crc_valid {
                flags |= LE_CRC_VALID;
            }
        }
        flags |= phy_to_flags(pkt.phy);

        let rf_channel = ((pkt.freq - 2402) / 2) as u8;
        let ble_payload_len = pkt.len + 10; // data + LE header (10 bytes)

        let ppi_len = if gps.map_or(false, |g| g.valid) {
            PPI_GPS_SIZE
        } else {
            PPI_HDR_SIZE
        };

        let total_len = ppi_len + ble_payload_len;

        // PCAP record header
        self.writer
            .write_u32::<LittleEndian>(pkt.timestamp.tv_sec as u32)?;
        self.writer
            .write_u32::<LittleEndian>((pkt.timestamp.tv_nsec / 1000) as u32)?;
        self.writer.write_u32::<LittleEndian>(total_len as u32)?;
        self.writer.write_u32::<LittleEndian>(total_len as u32)?;

        // PPI header
        write_ppi_header(
            &mut self.writer,
            ppi_len as u16,
            DLT_BLUETOOTH_LE_LL_WITH_PHDR,
        )?;

        // PPI GPS field (if present)
        if let Some(fix) = gps {
            if fix.valid {
                write_ppi_gps(&mut self.writer, fix)?;
            }
        }

        // BLE LE link-layer header (10 bytes, packed LE)
        self.writer.write_u8(rf_channel)?;
        self.writer.write_i8(pkt.rssi_db as i8)?;
        self.writer.write_i8(pkt.noise_db as i8)?;
        self.writer.write_u8(0)?; // aa_offenses
        self.writer.write_u32::<LittleEndian>(0)?; // ref_aa
        self.writer.write_u16::<LittleEndian>(flags)?;

        // Packet data
        self.writer.write_all(&pkt.data)?;
        self.writer.flush()?;
        Ok(())
    }

    /// Write a Classic BT packet record
    pub fn write_bt(&mut self, pkt: &ClassicBtPacket, gps: Option<&GpsFix>) -> io::Result<()> {
        let flags: u16 = BREDR_SIGNAL_POWER_VALID | BREDR_NOISE_POWER_VALID;

        let rf_channel = (pkt.freq - 2402) as u8;
        let payload_len = pkt.decoded_payload.len();
        let bt_hdr_len = 22usize; // pcap_bredr_bb_header_t size
        let total_bt_len = bt_hdr_len + payload_len;

        let ppi_len = if gps.map_or(false, |g| g.valid) {
            PPI_GPS_SIZE
        } else {
            PPI_HDR_SIZE
        };

        let total_len = ppi_len + total_bt_len;

        // PCAP record header
        self.writer
            .write_u32::<LittleEndian>(pkt.timestamp.tv_sec as u32)?;
        self.writer
            .write_u32::<LittleEndian>((pkt.timestamp.tv_nsec / 1000) as u32)?;
        self.writer.write_u32::<LittleEndian>(total_len as u32)?;
        self.writer.write_u32::<LittleEndian>(total_len as u32)?;

        // PPI header
        write_ppi_header(&mut self.writer, ppi_len as u16, DLT_BLUETOOTH_BREDR_BB)?;

        // PPI GPS field (if present)
        if let Some(fix) = gps {
            if fix.valid {
                write_ppi_gps(&mut self.writer, fix)?;
            }
        }

        // Classic BT BR/EDR baseband header (20 bytes, packed LE)
        self.writer.write_u8(rf_channel)?; // rf_channel
        self.writer.write_i8(pkt.rssi_db as i8)?; // signal_power
        self.writer.write_i8(pkt.noise_db as i8)?; // noise_power
        self.writer.write_u8(pkt.ac_errors)?; // access_code_offenses
        self.writer.write_u8(bredr_transport_rate(pkt))?;
        self.writer.write_u8(0)?; // corrected_header_bits
        self.writer.write_i16::<LittleEndian>(-1)?; // corrected_payload_bits
        let (ref_lap_uap, bt_header, addr_flags) = bredr_addr_fields(pkt);
        self.writer.write_u32::<LittleEndian>(pkt.lap)?; // lap
        self.writer.write_u32::<LittleEndian>(ref_lap_uap)?; // ref_lap_uap
        self.writer.write_u32::<LittleEndian>(bt_header)?; // bt_header
        self.writer
            .write_u16::<LittleEndian>(flags | addr_flags | bredr_payload_flags(pkt))?; // flags

        // LINKTYPE_BLUETOOTH_BREDR_BB stores the decoded header above and only
        // the decoded baseband payload after the pseudoheader.
        if !pkt.decoded_payload.is_empty() {
            self.writer.write_all(&pkt.decoded_payload)?;
        }
        self.writer.flush()?;
        Ok(())
    }

    /// Flush the writer
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Get inner writer
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// Build a ZMQ message buffer for a BLE packet.
/// Format: type_byte(1) + pcaprec_hdr(16) + le_header(10) + data
pub fn zmq_build_ble(pkt: &BlePacket) -> Vec<u8> {
    let mut flags: u16 = LE_DEWHITENED | LE_SIGNAL_POWER_VALID | LE_NOISE_POWER_VALID;
    if pkt.crc_checked {
        flags |= LE_CRC_CHECKED;
        if pkt.crc_valid {
            flags |= LE_CRC_VALID;
        }
    }
    flags |= phy_to_flags(pkt.phy);

    let ble_payload_len = pkt.len + 10; // data + LE header
    let msg_len = 1 + 16 + 10 + pkt.len;
    let mut buf = Vec::with_capacity(msg_len);

    // Type byte
    buf.push(ZMQ_PKT_TYPE_BLE);

    // pcaprec_hdr (16 bytes)
    buf.extend_from_slice(&(pkt.timestamp.tv_sec as u32).to_le_bytes());
    buf.extend_from_slice(&((pkt.timestamp.tv_nsec / 1000) as u32).to_le_bytes());
    buf.extend_from_slice(&(ble_payload_len as u32).to_le_bytes());
    buf.extend_from_slice(&(ble_payload_len as u32).to_le_bytes());

    // LE header (10 bytes)
    let rf_channel = ((pkt.freq - 2402) / 2) as u8;
    buf.push(rf_channel);
    buf.push(pkt.rssi_db as u8);
    buf.push(pkt.noise_db as u8);
    buf.push(0); // aa_offenses
    buf.extend_from_slice(&0u32.to_le_bytes()); // ref_aa
    buf.extend_from_slice(&flags.to_le_bytes());

    // Packet data
    buf.extend_from_slice(&pkt.data);

    buf
}

/// Build a ZMQ message buffer for a Classic BT packet.
/// Format: type_byte(1) + pcaprec_hdr(16) + bredr_header(22) + [decoded payload]
pub fn zmq_build_bt(pkt: &ClassicBtPacket) -> Vec<u8> {
    let flags: u16 = BREDR_SIGNAL_POWER_VALID | BREDR_NOISE_POWER_VALID;
    let payload_len = pkt.decoded_payload.len();
    let total_len = 22 + payload_len;
    let msg_len = 1 + 16 + total_len;
    let mut buf = Vec::with_capacity(msg_len);

    // Type byte
    buf.push(ZMQ_PKT_TYPE_BT);

    // pcaprec_hdr (16 bytes)
    buf.extend_from_slice(&(pkt.timestamp.tv_sec as u32).to_le_bytes());
    buf.extend_from_slice(&((pkt.timestamp.tv_nsec / 1000) as u32).to_le_bytes());
    buf.extend_from_slice(&(total_len as u32).to_le_bytes());
    buf.extend_from_slice(&(total_len as u32).to_le_bytes());

    // BR/EDR header (20 bytes)
    let rf_channel = (pkt.freq - 2402) as u8;
    buf.push(rf_channel);
    buf.push(pkt.rssi_db as u8);
    buf.push(pkt.noise_db as u8);
    buf.push(pkt.ac_errors);
    buf.push(bredr_transport_rate(pkt));
    buf.push(0); // corrected_header_bits
    buf.extend_from_slice(&(-1i16).to_le_bytes()); // corrected_payload_bits
    let (ref_lap_uap, bt_header, addr_flags) = bredr_addr_fields(pkt);
    buf.extend_from_slice(&pkt.lap.to_le_bytes());
    buf.extend_from_slice(&ref_lap_uap.to_le_bytes()); // ref_lap_uap
    buf.extend_from_slice(&bt_header.to_le_bytes()); // bt_header
    buf.extend_from_slice(&(flags | addr_flags | bredr_payload_flags(pkt)).to_le_bytes());

    if !pkt.decoded_payload.is_empty() {
        buf.extend_from_slice(&pkt.decoded_payload);
    }

    buf
}

/// Build a ZMQ GPS frame (24 bytes: 3x f64 LE)
pub fn zmq_build_gps_frame(fix: &GpsFix) -> [u8; 24] {
    let mut frame = [0u8; 24];
    frame[0..8].copy_from_slice(&fix.latitude.to_le_bytes());
    frame[8..16].copy_from_slice(&fix.longitude.to_le_bytes());
    frame[16..24].copy_from_slice(&fix.altitude.to_le_bytes());
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classic_packet(decoded_payload: Vec<u8>) -> ClassicBtPacket {
        ClassicBtPacket {
            lap: 0x123456,
            ac_errors: 0,
            sync_offset: 0,
            freq: 2437,
            rssi_db: -40,
            noise_db: -90,
            timestamp: bd_protocol::Timespec {
                tv_sec: 1000,
                tv_nsec: 500_000_000,
            },
            raw_header: [0xaa; 7],
            has_header: true,
            payload: Vec::new(),
            uap: Some(0x78),
            uap_verified: true,
            nap: None,
            header: Some(bd_protocol::btbb::BtHeader {
                lt_addr: 1,
                pkt_type: 4,
                flow: 1,
                arqn: 0,
                seqn: 1,
                hec: 0x9a,
                clk6: 17,
            }),
            decoded_payload,
            crc_ok: true,
            clkn: None,
        }
    }

    #[test]
    fn test_pcap_global_header() {
        let mut buf = Vec::new();
        let _writer = PcapWriter::new(&mut buf).unwrap();

        // Check PCAP magic
        assert_eq!(buf[0..4], [0xd4, 0xc3, 0xb2, 0xa1]);
        // Version 2.4
        assert_eq!(buf[4..6], [2, 0]);
        assert_eq!(buf[6..8], [4, 0]);
        // DLT_PPI = 192
        assert_eq!(buf[20..24], [192, 0, 0, 0]);
    }

    #[test]
    fn test_ppi_fixed_point() {
        // Test latitude 0 degrees -> fixed3_7 = (0 + 180) * 1e7 = 1_800_000_000
        assert_eq!(ppi_fixed3_7(0.0), 1_800_000_000);
        // Test altitude 0m -> fixed6_4 = (0 + 180000) * 1e4 = 1_800_000_000
        assert_eq!(ppi_fixed6_4(0.0), 1_800_000_000);
    }

    #[test]
    fn test_zmq_ble_message_format() {
        let pkt = BlePacket {
            aa: 0x8E89BED6,
            rssi_db: -40,
            noise_db: -90,
            freq: 2426,
            len: 9,
            timestamp: bd_protocol::Timespec {
                tv_sec: 1000,
                tv_nsec: 500_000_000,
            },
            crc_checked: true,
            crc_valid: true,
            is_data: false,
            conn_valid: false,
            phy: BlePhy::Phy1M,
            ext_header: None,
            data: vec![0xD6, 0xBE, 0x89, 0x8E, 0x40, 0x06, 0x01, 0x02, 0x03],
        };

        let msg = zmq_build_ble(&pkt);
        assert_eq!(msg[0], ZMQ_PKT_TYPE_BLE);
        // Total length in pcaprec_hdr should be 10 (LE header) + 9 (data) = 19
        let incl_len = u32::from_le_bytes(msg[9..13].try_into().unwrap());
        assert_eq!(incl_len, 19);
    }

    #[test]
    fn test_zmq_bt_omits_raw_coded_header() {
        let msg = zmq_build_bt(&classic_packet(Vec::new()));
        let incl_len = u32::from_le_bytes(msg[9..13].try_into().unwrap());
        let flags = u16::from_le_bytes(msg[37..39].try_into().unwrap());

        assert_eq!(incl_len, 22);
        assert_eq!(msg.len(), 1 + 16 + 22);
        assert_eq!(
            flags & (BREDR_SIGNAL_POWER_VALID | BREDR_NOISE_POWER_VALID),
            0x0006
        );
        assert_eq!(flags & (BREDR_DEWHITENED | BREDR_PAYLOAD_PRESENT), 0);
    }

    #[test]
    fn test_zmq_bt_appends_only_decoded_payload() {
        let payload = vec![0x0d, 0x01, 0x02, 0x03];
        let msg = zmq_build_bt(&classic_packet(payload.clone()));
        let incl_len = u32::from_le_bytes(msg[9..13].try_into().unwrap());
        let flags = u16::from_le_bytes(msg[37..39].try_into().unwrap());

        assert_eq!(incl_len, 22 + payload.len() as u32);
        assert_eq!(&msg[msg.len() - payload.len()..], payload.as_slice());
        assert_ne!(flags & BREDR_DEWHITENED, 0);
        assert_ne!(flags & BREDR_PAYLOAD_PRESENT, 0);
        assert_ne!(flags & BREDR_CRC_CHECKED, 0);
        assert_ne!(flags & BREDR_CRC_VALID, 0);
    }

    #[test]
    fn test_zmq_bt_does_not_mark_candidate_uap_valid() {
        let mut pkt = classic_packet(Vec::new());
        pkt.uap_verified = false;
        let msg = zmq_build_bt(&pkt);
        let flags = u16::from_le_bytes(msg[37..39].try_into().unwrap());

        assert_eq!(flags & BREDR_REFUAP_VALID, 0);
        assert_eq!(flags & (BREDR_HEC_CHECKED | BREDR_HEC_VALID), 0);
    }

    #[test]
    fn test_zmq_bt_marks_edr_transport_rate() {
        let mut pkt = classic_packet(vec![0x01, 0x02]);
        pkt.header.as_mut().unwrap().pkt_type = 0x6;
        let msg = zmq_build_bt(&pkt);

        assert_eq!(msg[21], 0x31);
    }

    #[test]
    fn test_zmq_bt_dashboard_contract_fixture() {
        let mut pkt = classic_packet(vec![0x01, 0x02, 0x03]);
        pkt.lap = 0x405060;
        pkt.uap = Some(0x30);
        pkt.freq = 2441;
        pkt.timestamp.tv_sec = 1;
        pkt.timestamp.tv_nsec = 0;
        pkt.header = Some(bd_protocol::btbb::BtHeader {
            lt_addr: 1,
            pkt_type: 0x0d,
            flow: 1,
            arqn: 0,
            seqn: 1,
            hec: 0xa5,
            clk6: 0,
        });

        let expected = b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x19\x00\x00\x00\x19\x00\x00\x00\
                         \x27\xd8\xa6\x00\x32\x00\xff\xff\x60\x50\x40\x00\x60\x50\x40\x30\
                         \xe9\x96\x02\x00\xb7\x0f\x01\x02\x03";
        assert_eq!(zmq_build_bt(&pkt), expected);
    }
}
