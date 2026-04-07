// Copyright 2026 CEMAXECUTER LLC
//
// WHAD (Wireless Hacking and Auditing Devices) native Rust integration.
//
// Communicates with WHAD-compatible devices (e.g., ButteRFly nRF52840 dongle)
// over serial/USB CDC-ACM using the WHAD protobuf protocol.
//
// Supports: BLE advertisement sniffing, raw PDU injection, connection sniffing,
// jamming, and connection hijacking.
//
// Wire format: 0xAC 0xBE (magic) + 2-byte LE length + protobuf payload.
// Protocol: https://github.com/whad-team/whad-protocol (MIT License)

use std::io::{Read, Write};
use std::time::Duration;

use prost::Message as ProstMessage;

use crate::whad_proto::{ble, discovery, generic, Message};
use crate::whad_proto::root::message::Msg as WhadMsg;

const WHAD_MAGIC: [u8; 2] = [0xAC, 0xBE];
const WHAD_PROTO_VERSION: u32 = 2;
const BLE_DOMAIN: u32 = 0x03000000;

// BLE command bit positions for supported_commands
const CMD_SNIFF_ADV: u64 = 1 << 0x01;
const CMD_JAM_ADV: u64 = 1 << 0x02;
const CMD_SNIFF_CONN_REQ: u64 = 1 << 0x05;
const CMD_SNIFF_ACTIVE_CONN: u64 = 1 << 0x07;
const CMD_SEND_RAW_PDU: u64 = 1 << 0x0e;
const CMD_START: u64 = 1 << 0x12;
const CMD_STOP: u64 = 1 << 0x13;
const CMD_HIJACK_MASTER: u64 = 1 << 0x15;
const CMD_HIJACK_SLAVE: u64 = 1 << 0x16;

/// Device information from WHAD discovery.
#[derive(Debug, Clone)]
pub struct WhadDeviceInfo {
    pub device_type: u32,
    pub fw_version: String,
    pub ble_commands: u64,
    pub capabilities: Vec<u32>,
}

impl WhadDeviceInfo {
    pub fn supports(&self, cmd_bit: u64) -> bool {
        self.ble_commands & cmd_bit != 0
    }

    pub fn supports_sniff_adv(&self) -> bool { self.supports(CMD_SNIFF_ADV) }
    pub fn supports_inject(&self) -> bool { self.supports(CMD_SEND_RAW_PDU) }
    pub fn supports_jam(&self) -> bool { self.supports(CMD_JAM_ADV) }
    pub fn supports_hijack(&self) -> bool {
        self.supports(CMD_HIJACK_MASTER) || self.supports(CMD_HIJACK_SLAVE)
    }
}

/// A received BLE packet from the WHAD device.
#[derive(Debug, Clone)]
pub struct WhadBlePacket {
    pub channel: u32,
    pub rssi: Option<i32>,
    pub timestamp: Option<u64>,
    pub access_address: u32,
    pub pdu: Vec<u8>,
    pub crc: u32,
    pub crc_valid: Option<bool>,
    pub direction: i32,
}

/// WHAD device handle for BLE operations.
pub struct WhadHandle {
    port: Box<dyn serialport::SerialPort>,
    read_buf: Vec<u8>,
    read_pos: usize,
    read_len: usize,
    pub info: Option<WhadDeviceInfo>,
}

impl WhadHandle {
    /// Open a WHAD device on the given serial port path.
    /// e.g., "/dev/ttyACM1" or "whad:/dev/ttyACM1"
    pub fn open(port_path: &str) -> Result<Self, String> {
        let path = port_path
            .strip_prefix("whad:")
            .unwrap_or(port_path);

        let port = serialport::new(path, 115200)
            .timeout(Duration::from_millis(500))
            .open()
            .map_err(|e| format!("failed to open {}: {}", path, e))?;

        eprintln!("WHAD: opened {}", path);

        let mut handle = Self {
            port,
            read_buf: vec![0u8; 8192],
            read_pos: 0,
            read_len: 0,
            info: None,
        };

        handle.discover()?;
        Ok(handle)
    }

    /// Discover device capabilities.
    fn discover(&mut self) -> Result<(), String> {
        // Query device info
        let msg = Message {
            msg: Some(WhadMsg::Discovery(
                discovery::Message {
                    msg: Some(discovery::message::Msg::InfoQuery(
                        discovery::DeviceInfoQuery {
                            proto_ver: WHAD_PROTO_VERSION,
                        },
                    )),
                },
            )),
        };
        self.send_message(&msg)?;

        // Wait for DeviceInfoResp
        let resp = self.recv_message_timeout(Duration::from_secs(3))?;
        let device_info = match resp.msg {
            Some(WhadMsg::Discovery(disc)) => {
                match disc.msg {
                    Some(discovery::message::Msg::InfoResp(info)) => info,
                    _ => return Err("unexpected response to DeviceInfoQuery".into()),
                }
            }
            _ => return Err("unexpected response type".into()),
        };

        let fw_ver = format!(
            "{}.{}.{}",
            device_info.fw_version_major,
            device_info.fw_version_minor,
            device_info.fw_version_rev
        );
        eprintln!(
            "WHAD: device type={}, firmware={}, capabilities={:?}",
            device_info.r#type, fw_ver, device_info.capabilities
        );

        // Query BLE domain capabilities
        let domain_query = Message {
            msg: Some(WhadMsg::Discovery(
                discovery::Message {
                    msg: Some(discovery::message::Msg::DomainQuery(
                        discovery::DeviceDomainInfoQuery {
                            domain: BLE_DOMAIN,
                        },
                    )),
                },
            )),
        };
        self.send_message(&domain_query)?;

        let domain_resp = self.recv_message_timeout(Duration::from_secs(3))?;
        let ble_commands = match domain_resp.msg {
            Some(WhadMsg::Discovery(disc)) => {
                match disc.msg {
                    Some(discovery::message::Msg::DomainResp(dr)) => dr.supported_commands,
                    _ => 0,
                }
            }
            _ => 0,
        };

        eprintln!("WHAD: BLE supported_commands=0x{:016X}", ble_commands);

        self.info = Some(WhadDeviceInfo {
            device_type: device_info.r#type,
            fw_version: fw_ver,
            ble_commands,
            capabilities: device_info.capabilities,
        });

        Ok(())
    }

    /// Start sniffing BLE advertisements on a specific channel (37, 38, or 39).
    /// Use channel=0xFF to listen on all advertising channels.
    pub fn sniff_adv(&mut self, channel: u32) -> Result<(), String> {
        let msg = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::SniffAdv(ble::SniffAdvCmd {
                    use_extended_adv: false,
                    channel,
                    bd_address: vec![0xFF; 6], // all devices
                })),
            })),
        };
        self.send_message(&msg)?;
        self.expect_success()?;

        // Send Start command
        let start = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::Start(ble::StartCmd {})),
            })),
        };
        self.send_message(&start)?;
        self.expect_success()?;

        eprintln!("WHAD: sniffing advertisements on channel {}", channel);
        Ok(())
    }

    /// Stop current operation.
    pub fn stop(&mut self) -> Result<(), String> {
        let msg = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::Stop(ble::StopCmd {})),
            })),
        };
        self.send_message(&msg)?;
        // Don't require success -- device may already be stopped
        Ok(())
    }

    /// Start peripheral mode: advertise with the given advertisement data.
    /// Uses PeripheralMode command (supported by ButteRFly, unlike AdvMode).
    pub fn start_peripheral(&mut self, adv_data: &[u8], scan_rsp: &[u8]) -> Result<(), String> {
        let msg = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::PeriphMode(ble::PeripheralModeCmd {
                    scan_data: adv_data.to_vec(),
                    scanrsp_data: scan_rsp.to_vec(),
                })),
            })),
        };
        self.send_message(&msg)?;
        self.expect_success()?;

        let start = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::Start(ble::StartCmd {})),
            })),
        };
        self.send_message(&start)?;
        self.expect_success()?;

        eprintln!("WHAD: peripheral mode started ({} bytes adv data)", adv_data.len());
        Ok(())
    }

    /// Send a raw BLE PDU on a specific channel.
    pub fn send_raw_pdu(
        &mut self,
        access_address: u32,
        pdu: &[u8],
        crc: u32,
    ) -> Result<(), String> {
        let msg = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::SendRawPdu(ble::SendRawPduCmd {
                    direction: ble::BleDirection::InjectionToSlave as i32,
                    conn_handle: 0,
                    access_address,
                    pdu: pdu.to_vec(),
                    crc,
                    encrypt: false,
                })),
            })),
        };
        self.send_message(&msg)?;
        self.expect_success()?;
        Ok(())
    }

    /// Sniff connection requests (CONNECT_IND) and follow the connection.
    pub fn sniff_conn_req(&mut self, target_mac: Option<&[u8; 6]>) -> Result<(), String> {
        let bd_addr = target_mac
            .map(|m| m.to_vec())
            .unwrap_or_else(|| vec![0xFF; 6]);

        let msg = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::SniffConnreq(ble::SniffConnReqCmd {
                    show_empty_packets: false,
                    show_advertisements: true,
                    channel: 0,
                    bd_address: bd_addr,
                })),
            })),
        };
        self.send_message(&msg)?;
        self.expect_success()?;

        let start = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::Start(ble::StartCmd {})),
            })),
        };
        self.send_message(&start)?;
        self.expect_success()?;

        eprintln!("WHAD: sniffing connection requests");
        Ok(())
    }

    /// Jam BLE advertisements on all channels.
    pub fn jam_adv(&mut self) -> Result<(), String> {
        let msg = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::JamAdv(ble::JamAdvCmd {})),
            })),
        };
        self.send_message(&msg)?;
        self.expect_success()?;

        let start = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::Start(ble::StartCmd {})),
            })),
        };
        self.send_message(&start)?;
        self.expect_success()?;

        eprintln!("WHAD: jamming advertisements");
        Ok(())
    }

    /// Jam BLE advertisements on a specific channel.
    pub fn jam_adv_channel(&mut self, channel: u32) -> Result<(), String> {
        let msg = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::JamAdvChan(ble::JamAdvOnChannelCmd {
                    channel,
                })),
            })),
        };
        self.send_message(&msg)?;
        self.expect_success()?;

        let start = Message {
            msg: Some(WhadMsg::Ble(ble::Message {
                msg: Some(ble::message::Msg::Start(ble::StartCmd {})),
            })),
        };
        self.send_message(&start)?;
        self.expect_success()?;

        eprintln!("WHAD: jamming channel {}", channel);
        Ok(())
    }

    /// Receive the next BLE packet (blocking with timeout).
    pub fn recv_packet(&mut self, timeout: Duration) -> Option<WhadBlePacket> {
        let msg = self.recv_message_timeout(timeout).ok()?;
        match msg.msg {
            Some(WhadMsg::Ble(ble_msg)) => {
                match ble_msg.msg {
                    Some(ble::message::Msg::RawPdu(raw)) => Some(WhadBlePacket {
                        channel: raw.channel,
                        rssi: raw.rssi,
                        timestamp: raw.timestamp,
                        access_address: raw.access_address,
                        pdu: raw.pdu,
                        crc: raw.crc,
                        crc_valid: raw.crc_validity,
                        direction: raw.direction,
                    }),
                    Some(ble::message::Msg::AdvPdu(adv)) => Some(WhadBlePacket {
                        channel: 0,
                        rssi: Some(adv.rssi),
                        timestamp: None,
                        access_address: 0x8E89BED6,
                        pdu: adv.adv_data,
                        crc: 0,
                        crc_valid: None,
                        direction: 0,
                    }),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    // -- Wire protocol --

    fn send_message(&mut self, msg: &Message) -> Result<(), String> {
        let payload = msg.encode_to_vec();
        let len = payload.len() as u16;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&WHAD_MAGIC);
        frame.extend_from_slice(&len.to_le_bytes());
        frame.extend_from_slice(&payload);
        self.port
            .write_all(&frame)
            .map_err(|e| format!("WHAD write error: {}", e))?;
        Ok(())
    }

    fn recv_message_timeout(&mut self, timeout: Duration) -> Result<Message, String> {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            if std::time::Instant::now() > deadline {
                return Err("WHAD recv timeout".into());
            }

            // Try to find magic bytes in buffer
            if let Some(msg) = self.try_parse_message() {
                return Ok(msg);
            }

            // Read more data
            self.port
                .set_timeout(Duration::from_millis(100))
                .ok();
            match self.port.read(&mut self.read_buf[self.read_len..]) {
                Ok(n) if n > 0 => {
                    self.read_len += n;
                }
                Ok(_) => continue,
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => continue,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(e) => return Err(format!("WHAD read error: {}", e)),
            }
        }
    }

    fn try_parse_message(&mut self) -> Option<Message> {
        let buf = &self.read_buf[self.read_pos..self.read_len];
        if buf.len() < 4 {
            return None;
        }

        // Find magic bytes
        for i in 0..buf.len() - 1 {
            if buf[i] == WHAD_MAGIC[0] && buf[i + 1] == WHAD_MAGIC[1] {
                let header_start = i;
                if header_start + 4 > buf.len() {
                    // Need more data for length
                    self.read_pos += header_start;
                    self.compact_buffer();
                    return None;
                }

                let len = u16::from_le_bytes([buf[header_start + 2], buf[header_start + 3]]) as usize;
                let msg_end = header_start + 4 + len;

                if msg_end > buf.len() {
                    // Need more data for payload
                    self.read_pos += header_start;
                    self.compact_buffer();
                    return None;
                }

                let payload = &buf[header_start + 4..msg_end];
                match Message::decode(payload) {
                    Ok(msg) => {
                        self.read_pos += msg_end;
                        self.compact_buffer();
                        return Some(msg);
                    }
                    Err(_) => {
                        // Bad message, skip past magic and try again
                        self.read_pos += header_start + 2;
                        continue;
                    }
                }
            }
        }

        // No magic found, discard searched bytes
        if buf.len() > 1 {
            self.read_pos = self.read_len - 1;
            self.compact_buffer();
        }
        None
    }

    fn compact_buffer(&mut self) {
        if self.read_pos > 0 {
            let remaining = self.read_len - self.read_pos;
            self.read_buf.copy_within(self.read_pos..self.read_len, 0);
            self.read_pos = 0;
            self.read_len = remaining;
        }
    }

    fn expect_success(&mut self) -> Result<(), String> {
        let resp = self.recv_message_timeout(Duration::from_secs(2))?;
        match resp.msg {
            Some(WhadMsg::Generic(gen)) => {
                match gen.msg {
                    Some(generic::message::Msg::CmdResult(result)) => {
                        if result.result == generic::ResultCode::Success as i32 {
                            Ok(())
                        } else {
                            Err(format!("WHAD command failed: result={}", result.result))
                        }
                    }
                    Some(generic::message::Msg::Result(code)) => {
                        if code == generic::ResultCode::Success as i32 {
                            Ok(())
                        } else {
                            Err(format!("WHAD command failed: code={}", code))
                        }
                    }
                    _ => Err("unexpected generic response".into()),
                }
            }
            _ => {
                // Some commands return BLE-specific responses instead of generic
                Ok(())
            }
        }
    }
}

impl Drop for WhadHandle {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
