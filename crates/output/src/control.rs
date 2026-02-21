// Copyright 2025-2026 CEMAXECUTER LLC

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam::channel::Sender;
use serde_json::json;

use crate::zmq_pub::parse_curve_keyfile;

/// Commands dispatched from the C2 control thread to the pipeline.
#[derive(Debug)]
pub enum ControlCommand {
    SetGain {
        gain: f64,
        lna: Option<i32>,
        vga: Option<i32>,
        req_id: Option<String>,
    },
    SetSquelch {
        threshold: f32,
        req_id: Option<String>,
    },
    Restart {
        center_freq: Option<u32>,
        channels: Option<u32>,
        req_id: Option<String>,
    },
}

/// Shared state for heartbeat reporting.
/// Updated by the pipeline thread, read by the control thread.
pub struct HeartbeatState {
    pub sensor_id: String,
    pub sdr_type: String,
    pub center_freq: u32,
    pub channels: u32,
    pub gain: f64,
    pub squelch: f32,
    pub total_pkts: u64,
    pub pkt_rate: f64,
    pub crc_pct: f64,
    pub gps: Option<(f64, f64, f64)>, // lat, lon, alt
}

impl HeartbeatState {
    pub fn new(sensor_id: &str, sdr_type: &str, center_freq: u32, channels: u32) -> Self {
        Self {
            sensor_id: sensor_id.to_string(),
            sdr_type: sdr_type.to_string(),
            center_freq,
            channels,
            gain: 0.0,
            squelch: -45.0,
            total_pkts: 0,
            pkt_rate: 0.0,
            crc_pct: 0.0,
            gps: None,
        }
    }
}

/// C2 control module: DEALER socket connecting to dashboard ROUTER.
pub struct ControlClient {
    socket: zmq::Socket,
    _ctx: zmq::Context,
    state: Arc<Mutex<HeartbeatState>>,
    cmd_tx: Sender<ControlCommand>,
    running: Arc<AtomicBool>,
    start_time: Instant,
}

impl ControlClient {
    /// Create a new DEALER socket and connect to the control endpoint.
    pub fn new(
        control_endpoint: &str,
        sensor_id: &str,
        curve_keyfile: Option<&str>,
        state: Arc<Mutex<HeartbeatState>>,
        cmd_tx: Sender<ControlCommand>,
        running: Arc<AtomicBool>,
    ) -> Result<Self, String> {
        let ctx = zmq::Context::new();
        let socket = ctx
            .socket(zmq::DEALER)
            .map_err(|e| format!("zmq DEALER socket: {}", e))?;

        // Set identity so ROUTER can route replies back
        socket
            .set_identity(sensor_id.as_bytes())
            .map_err(|e| format!("zmq set_identity: {}", e))?;

        socket
            .set_sndhwm(100)
            .map_err(|e| format!("zmq set_sndhwm: {}", e))?;
        socket
            .set_rcvhwm(100)
            .map_err(|e| format!("zmq set_rcvhwm: {}", e))?;

        // CurveZMQ: DEALER is CURVE client, ROUTER is CURVE server
        if let Some(keyfile) = curve_keyfile {
            let (server_public_key, _server_secret_key) = parse_curve_keyfile(keyfile)?;

            // Generate ephemeral client keypair
            let client_keypair = zmq::CurveKeyPair::new()
                .map_err(|e| format!("zmq curve_keypair: {}", e))?;

            socket
                .set_curve_serverkey(server_public_key.as_bytes())
                .map_err(|e| format!("zmq curve_serverkey: {}", e))?;
            socket
                .set_curve_publickey(&client_keypair.public_key)
                .map_err(|e| format!("zmq curve_publickey: {}", e))?;
            socket
                .set_curve_secretkey(&client_keypair.secret_key)
                .map_err(|e| format!("zmq curve_secretkey: {}", e))?;
        }

        socket
            .connect(control_endpoint)
            .map_err(|e| format!("zmq connect to {}: {}", control_endpoint, e))?;

        eprintln!("C2: connected to {}", control_endpoint);

        Ok(Self {
            socket,
            _ctx: ctx,
            state,
            cmd_tx,
            running,
            start_time: Instant::now(),
        })
    }

    /// Run the control loop (blocking). Call from a dedicated thread.
    pub fn run(&self) {
        // Send initial heartbeat
        self.send_heartbeat();
        let mut last_heartbeat = Instant::now();

        while self.running.load(Ordering::Relaxed) {
            // Poll for incoming commands (1 second timeout)
            if let Ok(events) = self.socket.poll(zmq::POLLIN, 1000) {
                if events > 0 {
                    if let Ok(msg) = self.socket.recv_bytes(0) {
                        self.dispatch_command(&msg);
                    }
                }
            }

            // Send heartbeat every 5 seconds
            if last_heartbeat.elapsed().as_secs() >= 5 {
                self.send_heartbeat();
                last_heartbeat = Instant::now();
            }
        }
    }

    fn send_heartbeat(&self) {
        let state = self.state.lock().unwrap();
        let uptime = self.start_time.elapsed().as_secs();

        let mut hb = json!({
            "type": "heartbeat",
            "sensor_id": state.sensor_id,
            "sdr": state.sdr_type,
            "center_freq": state.center_freq,
            "channels": state.channels,
            "gain": { "value": state.gain },
            "squelch": state.squelch,
            "total_pkts": state.total_pkts as f64,
            "pkt_rate": (state.pkt_rate * 10.0).round() / 10.0,
            "crc_pct": (state.crc_pct * 10.0).round() / 10.0,
            "uptime": uptime,
        });

        if let Some((lat, lon, alt)) = state.gps {
            hb["gps"] = json!([lat, lon, alt]);
        }

        drop(state);

        let json_str = hb.to_string();
        let _ = self.socket.send(&json_str, zmq::DONTWAIT);
    }

    fn send_response(&self, req_id: Option<&str>, status: &str, message: &str) {
        let mut resp = json!({
            "type": "response",
            "status": status,
            "message": message,
        });
        if let Some(id) = req_id {
            resp["req_id"] = json!(id);
        }
        let json_str = resp.to_string();
        let _ = self.socket.send(&json_str, zmq::DONTWAIT);
    }

    fn dispatch_command(&self, data: &[u8]) {
        let root: serde_json::Value = match serde_json::from_slice(data) {
            Ok(v) => v,
            Err(_) => return,
        };

        let cmd = match root.get("cmd").and_then(|c| c.as_str()) {
            Some(c) => c,
            None => return,
        };
        let req_id = root.get("req_id").and_then(|r| r.as_str());

        match cmd {
            "set_gain" => {
                let gain = root.get("gain").and_then(|g| g.as_f64()).unwrap_or(-1.0);
                let lna = root.get("lna").and_then(|l| l.as_i64()).map(|l| l as i32);
                let vga = root.get("vga").and_then(|v| v.as_i64()).map(|v| v as i32);

                let command = ControlCommand::SetGain {
                    gain,
                    lna,
                    vga,
                    req_id: req_id.map(|s| s.to_string()),
                };
                if self.cmd_tx.try_send(command).is_ok() {
                    self.send_response(req_id, "ok", "gain command queued");
                } else {
                    self.send_response(req_id, "error", "command queue full");
                }
            }
            "set_squelch" => {
                let threshold = match root.get("threshold").and_then(|t| t.as_f64()) {
                    Some(t) => t as f32,
                    None => {
                        self.send_response(req_id, "error", "missing threshold");
                        return;
                    }
                };
                if threshold > -5.0 || threshold < -100.0 {
                    self.send_response(req_id, "error", "threshold out of range (-100 to -5)");
                    return;
                }
                let command = ControlCommand::SetSquelch {
                    threshold,
                    req_id: req_id.map(|s| s.to_string()),
                };
                if self.cmd_tx.try_send(command).is_ok() {
                    self.send_response(req_id, "ok", &format!("squelch set to {:.1} dB", threshold));
                } else {
                    self.send_response(req_id, "error", "command queue full");
                }
            }
            "get_status" => {
                self.send_heartbeat();
            }
            "restart" => {
                let center_freq = root
                    .get("center_freq")
                    .and_then(|f| f.as_u64())
                    .map(|f| f as u32);
                let channels = root
                    .get("channels")
                    .and_then(|c| c.as_u64())
                    .map(|c| c as u32);

                if center_freq.is_none() && channels.is_none() {
                    self.send_response(
                        req_id,
                        "error",
                        "restart requires center_freq or channels",
                    );
                    return;
                }

                let command = ControlCommand::Restart {
                    center_freq,
                    channels,
                    req_id: req_id.map(|s| s.to_string()),
                };
                if self.cmd_tx.try_send(command).is_ok() {
                    self.send_response(req_id, "ok", "restarting");
                } else {
                    self.send_response(req_id, "error", "command queue full");
                }
            }
            _ => {
                self.send_response(req_id, "error", "unknown command");
            }
        }
    }
}
