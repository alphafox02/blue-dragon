// Copyright 2025-2026 CEMAXECUTER LLC

use std::fs;

use bd_protocol::ble::BlePacket;
use bd_protocol::btbb::ClassicBtPacket;

use crate::pcap::{self, GpsFix};

/// ZMQ publisher for streaming packets to dashboard.
/// Sensor PUB socket connects out; dashboard SUB socket binds.
pub struct ZmqPublisher {
    socket: zmq::Socket,
    sensor_id: Option<String>,
    _ctx: zmq::Context,
}

impl ZmqPublisher {
    /// Create a new ZMQ PUB socket and connect to the given endpoint.
    /// `curve_keyfile`: path to keyfile with public_key= and secret_key= lines.
    pub fn new(
        endpoint: &str,
        sensor_id: Option<&str>,
        curve_keyfile: Option<&str>,
    ) -> Result<Self, String> {
        let ctx = zmq::Context::new();
        let socket = ctx
            .socket(zmq::PUB)
            .map_err(|e| format!("zmq PUB socket: {}", e))?;

        socket
            .set_sndhwm(1000)
            .map_err(|e| format!("zmq set_sndhwm: {}", e))?;

        if let Some(keyfile) = curve_keyfile {
            let (public_key, secret_key) = parse_curve_keyfile(keyfile)?;
            socket
                .set_curve_server(true)
                .map_err(|e| format!("zmq curve_server: {}", e))?;
            socket
                .set_curve_secretkey(secret_key.as_bytes())
                .map_err(|e| format!("zmq curve_secretkey: {}", e))?;
            socket
                .set_curve_publickey(public_key.as_bytes())
                .map_err(|e| format!("zmq curve_publickey: {}", e))?;
            eprintln!(
                "ZMQ CURVE: encrypted (server key: {}...)",
                &public_key[..8.min(public_key.len())]
            );
        }

        socket
            .connect(endpoint)
            .map_err(|e| format!("zmq connect to {}: {}", endpoint, e))?;

        eprintln!("ZMQ PUB: connected to {}", endpoint);

        Ok(Self {
            socket,
            sensor_id: sensor_id.map(|s| s.to_string()),
            _ctx: ctx,
        })
    }

    /// Publish a BLE packet (multipart: [sensor_id] [gps] data)
    pub fn send_ble(&self, pkt: &BlePacket, gps: Option<&GpsFix>) {
        let buf = pcap::zmq_build_ble(pkt);
        self.send_raw(&buf, gps);
    }

    /// Publish a Classic BT packet (multipart: [sensor_id] [gps] data)
    pub fn send_bt(&self, pkt: &ClassicBtPacket, gps: Option<&GpsFix>) {
        let buf = pcap::zmq_build_bt(pkt);
        self.send_raw(&buf, gps);
    }

    /// Publish a GATT result as JSON on the "gatt:" topic.
    pub fn send_gatt(&self, result: &serde_json::Value) {
        // Topic frame: "gatt:"
        let _ = self.socket.send("gatt:", zmq::DONTWAIT | zmq::SNDMORE);
        if let Some(ref id) = self.sensor_id {
            let _ = self.socket.send(id.as_bytes(), zmq::DONTWAIT | zmq::SNDMORE);
        }
        let json_bytes = result.to_string();
        let _ = self.socket.send(json_bytes.as_bytes(), zmq::DONTWAIT);
    }

    /// Publish an active-scan result as JSON on the "scan:" topic.
    pub fn send_scan(&self, result: &serde_json::Value) {
        let _ = self.socket.send("scan:", zmq::DONTWAIT | zmq::SNDMORE);
        if let Some(ref id) = self.sensor_id {
            let _ = self.socket.send(id.as_bytes(), zmq::DONTWAIT | zmq::SNDMORE);
        }
        let json_bytes = result.to_string();
        let _ = self.socket.send(json_bytes.as_bytes(), zmq::DONTWAIT);
    }

    /// Send raw packet data with optional sensor_id and GPS frames.
    fn send_raw(&self, buf: &[u8], gps: Option<&GpsFix>) {
        if let Some(ref id) = self.sensor_id {
            let _ = self
                .socket
                .send(id.as_bytes(), zmq::DONTWAIT | zmq::SNDMORE);
        }
        if let Some(fix) = gps {
            if fix.valid {
                let gps_frame = pcap::zmq_build_gps_frame(fix);
                let _ = self.socket.send(&gps_frame[..], zmq::DONTWAIT | zmq::SNDMORE);
            }
        }
        let _ = self.socket.send(buf, zmq::DONTWAIT);
    }
}

impl Drop for ZmqPublisher {
    fn drop(&mut self) {
        // zmq::Socket::drop handles close; Context::drop handles ctx_destroy
    }
}

/// Parse a CurveZMQ keyfile (public_key=... and secret_key=... lines).
pub fn parse_curve_keyfile(path: &str) -> Result<(String, String), String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("failed to read curve keyfile {}: {}", path, e))?;

    let mut public_key = None;
    let mut secret_key = None;

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some(key) = line.strip_prefix("public_key=") {
            if key.len() >= 40 {
                public_key = Some(key[..40].to_string());
            }
        } else if let Some(key) = line.strip_prefix("secret_key=") {
            if key.len() >= 40 {
                secret_key = Some(key[..40].to_string());
            }
        }
    }

    match (public_key, secret_key) {
        (Some(pub_k), Some(sec_k)) => Ok((pub_k, sec_k)),
        _ => Err(format!("failed to parse curve keys from {}", path)),
    }
}

/// Derive control endpoint from data endpoint (port + 1).
pub fn derive_control_endpoint(data_endpoint: &str) -> String {
    if let Some(colon_pos) = data_endpoint.rfind(':') {
        let prefix = &data_endpoint[..=colon_pos];
        if let Ok(port) = data_endpoint[colon_pos + 1..].parse::<u16>() {
            return format!("{}{}", prefix, port + 1);
        }
    }
    data_endpoint.to_string()
}
