// Copyright 2025-2026 CEMAXECUTER LLC

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::pcap::GpsFix;

/// gpsd client that polls via TCP JSON protocol and caches the fix.
/// Runs a background thread to continuously read from gpsd.
pub struct GpsClient {
    cached: Arc<Mutex<GpsFix>>,
}

impl GpsClient {
    /// Connect to gpsd and start background reader thread.
    /// `host`: gpsd hostname (default "localhost")
    /// `port`: gpsd port (default 2947)
    pub fn new(host: &str, port: u16) -> Result<Self, String> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect_timeout(
            &addr
                .parse()
                .map_err(|e| format!("invalid gpsd address {}: {}", addr, e))?,
            Duration::from_secs(5),
        )
        .map_err(|e| format!("failed to connect to gpsd at {}: {}", addr, e))?;

        // Enable JSON watch mode
        let mut writer = stream
            .try_clone()
            .map_err(|e| format!("stream clone: {}", e))?;
        writer
            .write_all(b"?WATCH={\"enable\":true,\"json\":true};\n")
            .map_err(|e| format!("gpsd WATCH: {}", e))?;
        writer
            .flush()
            .map_err(|e| format!("gpsd flush: {}", e))?;

        let cached = Arc::new(Mutex::new(GpsFix::default()));
        let cached_bg = cached.clone();

        // Set a read timeout so the thread can check for shutdown
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .map_err(|e| format!("set_read_timeout: {}", e))?;

        // Background thread: read JSON lines from gpsd
        thread::Builder::new()
            .name("gpsd-reader".to_string())
            .spawn(move || {
                let reader = BufReader::new(stream);
                for line in reader.lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => continue,
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                        Err(e) => {
                            log::warn!("gpsd read error: {}", e);
                            break;
                        }
                    };

                    // Parse TPV (Time-Position-Velocity) class
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                        if val.get("class").and_then(|c| c.as_str()) == Some("TPV") {
                            let mode = val.get("mode").and_then(|m| m.as_i64()).unwrap_or(0);
                            if mode >= 2 {
                                let lat = val.get("lat").and_then(|v| v.as_f64());
                                let lon = val.get("lon").and_then(|v| v.as_f64());
                                let alt = val
                                    .get("altMSL")
                                    .or_else(|| val.get("alt"))
                                    .and_then(|v| v.as_f64());

                                if let (Some(lat), Some(lon)) = (lat, lon) {
                                    if lat.is_finite() && lon.is_finite() {
                                        let mut fix = cached_bg.lock().unwrap();
                                        fix.latitude = lat;
                                        fix.longitude = lon;
                                        fix.altitude = alt.filter(|a| a.is_finite()).unwrap_or(0.0);
                                        fix.valid = true;
                                    }
                                }
                            } else {
                                // No fix
                                let mut fix = cached_bg.lock().unwrap();
                                fix.valid = false;
                            }
                        }
                    }
                }
                log::info!("gpsd reader thread exiting");
            })
            .map_err(|e| format!("gpsd thread: {}", e))?;

        eprintln!("GPS: connected to gpsd at {}", addr);

        Ok(Self { cached })
    }

    /// Get the current GPS fix (cached, thread-safe).
    pub fn get_fix(&self) -> GpsFix {
        self.cached.lock().unwrap().clone()
    }
}
