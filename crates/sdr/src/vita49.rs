// Copyright 2026 CEMAXECUTER LLC
//
// VITA 49 (VRT) UDP input - receives IQ samples via VRT signal/context packets.
//
// Auto-detects sample rate, center frequency, and IQ format from VRT IF context
// packets (type 0x4). Falls back to user-supplied values if no context arrives.
//
// Pull-based: recv_into_i16() blocks until samples are available, matching the
// pattern used by USRP/HackRF/bladeRF backends.
//
// Usage: -i vita49 or -i vita49:192.168.1.100:5000
// Default bind: 0.0.0.0:4991

use std::net::UdpSocket;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_PORT: u16 = 4991;
const VRT_MAX_PKT: usize = 65536;

// VRT packet types
const VRT_TYPE_SIGNAL_DATA_NO_SID: u8 = 0x0;
const VRT_TYPE_SIGNAL_DATA: u8 = 0x1;
const VRT_TYPE_IF_CONTEXT: u8 = 0x4;
const VRT_TYPE_EXT_CONTEXT: u8 = 0x5;

// CIF0 field bit positions (VITA 49.0 Section 7.1.5)
const CIF_RF_REF_FREQ: u32 = 1 << 27;
const CIF_SAMPLE_RATE: u32 = 1 << 21;
const CIF_DATA_FORMAT: u32 = 1 << 15;

/// CIF0 field table: (bit flag, size in 32-bit words), MSB-first order.
const CIF_FIELDS: &[(u32, u32)] = &[
    (1 << 31, 1), // change indicator
    (1 << 30, 1), // reference point ID
    (1 << 29, 2), // bandwidth
    (1 << 28, 2), // IF reference frequency
    (CIF_RF_REF_FREQ, 2), // RF reference frequency
    (1 << 26, 2), // RF/IF frequency offset
    (1 << 25, 2), // IF band offset
    (1 << 24, 1), // reference level
    (1 << 23, 1), // gain
    (1 << 22, 1), // over-range count
    (CIF_SAMPLE_RATE, 2), // sample rate
    (1 << 20, 2), // timestamp adjustment
    (1 << 19, 1), // timestamp calibration time
    (1 << 18, 1), // temperature
    (1 << 17, 2), // device identifier
    (1 << 16, 1), // state/event indicators
    (CIF_DATA_FORMAT, 2), // data packet payload format
];

#[derive(Debug, Clone, Copy, PartialEq)]
enum IqFormat {
    Ci8,
    Ci16,
    Cf32,
}

pub struct Vita49Handle {
    sock: UdpSocket,
    pkt_buf: Vec<u8>,
    /// Residual i16 samples from previous packet not yet consumed
    residual: Vec<i16>,
    sample_rate: u32,
    center_freq: u64,
    iq_format: IqFormat,
    ctx_logged: bool,
    running: Arc<AtomicBool>,
    gap_count: AtomicU64,
    last_seq: u8,
    have_seq: bool,
}

impl Vita49Handle {
    /// Open a VITA 49 UDP receiver.
    ///
    /// `iface` format: "vita49", "vita49:PORT", or "vita49:IP:PORT"
    /// `sample_rate` and `center_freq_hz` are defaults; overridden by context packets.
    pub fn open(
        iface: &str,
        sample_rate: u32,
        center_freq_hz: u64,
        _gain: f64,
    ) -> Result<Self, String> {
        let (bind_addr, port) = parse_endpoint(iface)?;
        let bind = format!("{}:{}", bind_addr, port);

        let sock = UdpSocket::bind(&bind)
            .map_err(|e| format!("vita49: bind {}: {}", bind, e))?;

        // 500ms timeout so recv_into_i16 returns periodically for stop checks
        sock.set_read_timeout(Some(Duration::from_millis(500)))
            .map_err(|e| format!("vita49: set_read_timeout: {}", e))?;

        // 4 MB receive buffer for bursty traffic
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = sock.as_raw_fd();
            let buf_sz: libc::c_int = 4 * 1024 * 1024;
            unsafe {
                libc::setsockopt(
                    fd,
                    libc::SOL_SOCKET,
                    libc::SO_RCVBUF,
                    &buf_sz as *const _ as *const libc::c_void,
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                );
            }
        }

        eprintln!("vita49: listening on {}", bind);

        Ok(Self {
            sock,
            pkt_buf: vec![0u8; VRT_MAX_PKT],
            residual: Vec::with_capacity(8192),
            sample_rate,
            center_freq: center_freq_hz,
            iq_format: IqFormat::Ci16,
            ctx_logged: false,
            running: Arc::new(AtomicBool::new(true)),
            gap_count: AtomicU64::new(0),
            last_seq: 0,
            have_seq: false,
        })
    }

    /// Receive IQ samples into caller's i16 buffer. Returns number of complex
    /// samples written (buf filled with interleaved I,Q,I,Q,...).
    /// Blocks until at least one packet arrives or timeout.
    pub fn recv_into_i16(&mut self, buf: &mut [i16]) -> usize {
        let mut written = 0usize;

        // Drain residual first
        if !self.residual.is_empty() {
            let n = self.residual.len().min(buf.len());
            buf[..n].copy_from_slice(&self.residual[..n]);
            self.residual.drain(..n);
            written = n;
            if written >= buf.len() {
                return written / 2;
            }
        }

        // Keep receiving until buffer is full or timeout
        loop {
            let len = match self.sock.recv(&mut self.pkt_buf) {
                Ok(n) => n,
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    break;
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {
                    continue;
                }
                Err(_) => break,
            };

            let pkt = &self.pkt_buf[..len];

            // Copy packet to avoid borrow conflicts with self
            let vrl_off = vrl_strip(&self.pkt_buf[..len]);
            let vrt_len = len - vrl_off;
            if vrt_len < 4 {
                continue;
            }
            // Work from the original buffer using absolute offsets
            let vrt_start = vrl_off;

            let w0 = u32::from_be_bytes([
                self.pkt_buf[vrt_start],
                self.pkt_buf[vrt_start + 1],
                self.pkt_buf[vrt_start + 2],
                self.pkt_buf[vrt_start + 3],
            ]);
            let pkt_type = ((w0 >> 28) & 0xF) as u8;

            // Handle context packets
            if pkt_type == VRT_TYPE_IF_CONTEXT || pkt_type == VRT_TYPE_EXT_CONTEXT {
                if !self.ctx_logged {
                    let vrt_copy: Vec<u8> = self.pkt_buf[vrt_start..len].to_vec();
                    self.handle_context(&vrt_copy);
                }
                continue;
            }

            // Parse signal data
            let vrt_slice = &self.pkt_buf[vrt_start..len];
            let (payload_off, payload_bytes) = match parse_signal_header(vrt_slice) {
                Some(v) => v,
                None => continue,
            };

            // Sequence gap detection
            let count = ((w0 >> 16) & 0xF) as u8;
            if self.have_seq {
                let expected = (self.last_seq + 1) & 0xF;
                if count != expected {
                    self.gap_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.last_seq = count;
            self.have_seq = true;

            let payload = &self.pkt_buf[vrt_start + payload_off..vrt_start + payload_off + payload_bytes];

            // Convert to i16
            let samples = match self.iq_format {
                IqFormat::Ci8 => convert_ci8(payload),
                IqFormat::Ci16 => convert_ci16_be(payload),
                IqFormat::Cf32 => convert_cf32_be(payload),
            };

            if samples.is_empty() {
                continue;
            }

            // Copy as much as fits into buf, save rest as residual
            let space = buf.len() - written;
            let n = samples.len().min(space);
            buf[written..written + n].copy_from_slice(&samples[..n]);
            written += n;

            if n < samples.len() {
                self.residual.extend_from_slice(&samples[n..]);
            }

            if written >= buf.len() {
                break;
            }
        }

        written / 2 // return complex sample count
    }

    pub fn max_samps(&self) -> usize {
        // VRT packets are typically 1-8K samples; 16K is a safe max
        16384
    }

    pub fn overflow_count(&self) -> u64 {
        self.gap_count.load(Ordering::Relaxed)
    }

    pub fn set_gain(&self, _gain: f64) {
        // no-op: VITA 49 source controls gain externally
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn center_frequency(&self) -> u64 {
        self.center_freq
    }

    pub fn running(&self) -> &Arc<AtomicBool> {
        &self.running
    }

    fn handle_context(&mut self, vrt: &[u8]) {
        if let Some(ctx) = parse_context(vrt) {
            if let Some(sr) = ctx.sample_rate {
                self.sample_rate = sr as u32;
                eprintln!("vita49: auto-detected sample_rate={} Hz ({} Msps)",
                    sr as u64, sr as u64 / 1_000_000);
            }
            if let Some(freq) = ctx.center_freq {
                self.center_freq = freq as u64;
                eprintln!("vita49: auto-detected rf_freq={} Hz ({} MHz)",
                    freq as u64, freq as u64 / 1_000_000);
            }
            if let Some(fmt) = ctx.format {
                self.iq_format = fmt;
                eprintln!("vita49: auto-detected format={:?}", fmt);
            }
            self.ctx_logged = true;
        }
    }
}

/// Metadata auto-detected from VRT context packets
struct ContextInfo {
    sample_rate: Option<f64>,
    center_freq: Option<f64>,
    format: Option<IqFormat>,
}

/// Strip VRL wrapper ("VRLP" magic). Returns offset to VRT packet start.
fn vrl_strip(pkt: &[u8]) -> usize {
    if pkt.len() >= 12 {
        let magic = u32::from_be_bytes([pkt[0], pkt[1], pkt[2], pkt[3]]);
        if magic == 0x56524C50 {
            return 8;
        }
    }
    0
}

/// Parse VRT signal data header. Returns (payload_offset, payload_bytes).
fn parse_signal_header(vrt: &[u8]) -> Option<(usize, usize)> {
    if vrt.len() < 4 {
        return None;
    }

    let w0 = u32::from_be_bytes([vrt[0], vrt[1], vrt[2], vrt[3]]);
    let pkt_type = ((w0 >> 28) & 0xF) as u8;
    let class_id = (w0 >> 27) & 1;
    let trailer = (w0 >> 26) & 1;
    let tsi = (w0 >> 22) & 0x3;
    let tsf = (w0 >> 20) & 0x3;
    let pkt_size_w = (w0 & 0xFFFF) as usize;

    if pkt_type != VRT_TYPE_SIGNAL_DATA_NO_SID && pkt_type != VRT_TYPE_SIGNAL_DATA {
        return None;
    }

    let pkt_size_bytes = pkt_size_w * 4;
    if pkt_size_bytes > vrt.len() || pkt_size_bytes < 4 {
        return None;
    }

    let mut hdr_words: usize = 1;
    if pkt_type == VRT_TYPE_SIGNAL_DATA {
        hdr_words += 1;
    }
    if class_id != 0 {
        hdr_words += 2;
    }
    if tsi != 0 {
        hdr_words += 1;
    }
    if tsf != 0 {
        hdr_words += 2;
    }

    let trailer_words = if trailer != 0 { 1 } else { 0 };
    if hdr_words + trailer_words >= pkt_size_w {
        return None;
    }

    let payload_words = pkt_size_w - hdr_words - trailer_words;
    Some((hdr_words * 4, payload_words * 4))
}

/// Parse VRT IF context packet for sample rate, RF frequency, and data format.
fn parse_context(vrt: &[u8]) -> Option<ContextInfo> {
    if vrt.len() < 8 {
        return None;
    }

    let w0 = u32::from_be_bytes([vrt[0], vrt[1], vrt[2], vrt[3]]);
    let class_id = (w0 >> 27) & 1;
    let tsi = (w0 >> 22) & 0x3;
    let tsf = (w0 >> 20) & 0x3;
    let pkt_size_w = (w0 & 0xFFFF) as usize;

    if pkt_size_w * 4 > vrt.len() || pkt_size_w < 2 {
        return None;
    }

    let mut off_w: usize = 1;
    off_w += 1; // stream ID (always present in context)
    if class_id != 0 {
        off_w += 2;
    }
    if tsi != 0 {
        off_w += 1;
    }
    if tsf != 0 {
        off_w += 2;
    }

    if off_w >= pkt_size_w {
        return None;
    }

    let cif0 = read_be32(vrt, off_w);
    off_w += 1;

    let mut info = ContextInfo {
        sample_rate: None,
        center_freq: None,
        format: None,
    };

    for &(bit, words) in CIF_FIELDS {
        if cif0 & bit == 0 {
            continue;
        }
        if off_w + words as usize > pkt_size_w {
            return None;
        }

        if bit == CIF_RF_REF_FREQ {
            let val = read_be64(vrt, off_w);
            info.center_freq = Some(val as f64 / (1u64 << 20) as f64);
        } else if bit == CIF_SAMPLE_RATE {
            let val = read_be64(vrt, off_w);
            info.sample_rate = Some(val as f64 / (1u64 << 20) as f64);
        } else if bit == CIF_DATA_FORMAT {
            let fmt_w1 = read_be32(vrt, off_w);
            info.format = parse_data_format(fmt_w1);
        }

        off_w += words as usize;
    }

    if info.sample_rate.is_some() || info.center_freq.is_some() || info.format.is_some() {
        Some(info)
    } else {
        None
    }
}

fn parse_data_format(fmt_word1: u32) -> Option<IqFormat> {
    let real_cpx = (fmt_word1 >> 29) & 0x3;
    let dif = (fmt_word1 >> 24) & 0x1F;
    let data_bits = (fmt_word1 & 0x1F) + 1;

    if real_cpx != 1 {
        return None;
    }

    match dif {
        0 => match data_bits {
            8 => Some(IqFormat::Ci8),
            16 => Some(IqFormat::Ci16),
            _ => None,
        },
        14 => Some(IqFormat::Cf32),
        _ => None,
    }
}

fn convert_ci8(payload: &[u8]) -> Vec<i16> {
    payload.iter().map(|&b| (b as i8 as i16) << 8).collect()
}

fn convert_ci16_be(payload: &[u8]) -> Vec<i16> {
    let n = payload.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(i16::from_be_bytes([payload[i * 2], payload[i * 2 + 1]]));
    }
    out
}

fn convert_cf32_be(payload: &[u8]) -> Vec<i16> {
    let n = payload.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = i * 4;
        let f = f32::from_be_bytes([payload[b], payload[b + 1], payload[b + 2], payload[b + 3]]);
        out.push((f * 32767.0).clamp(-32768.0, 32767.0) as i16);
    }
    out
}

fn read_be32(data: &[u8], word_offset: usize) -> u32 {
    let off = word_offset * 4;
    u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

fn read_be64(data: &[u8], word_offset: usize) -> u64 {
    let hi = read_be32(data, word_offset) as u64;
    let lo = read_be32(data, word_offset + 1) as u64;
    (hi << 32) | lo
}

fn parse_endpoint(iface: &str) -> Result<(String, u16), String> {
    let rest = iface.strip_prefix("vita49").unwrap_or(iface);

    if rest.is_empty() {
        return Ok(("0.0.0.0".to_string(), DEFAULT_PORT));
    }

    let rest = rest.strip_prefix(':').ok_or_else(|| {
        format!("invalid vita49 endpoint: '{}' (use vita49, vita49:PORT, or vita49:IP:PORT)", iface)
    })?;

    if let Ok(port) = rest.parse::<u16>() {
        return Ok(("0.0.0.0".to_string(), port));
    }

    if let Some((ip, port_str)) = rest.rsplit_once(':') {
        let port = port_str
            .parse::<u16>()
            .map_err(|_| format!("invalid port in vita49 endpoint: '{}'", iface))?;
        return Ok((ip.to_string(), port));
    }

    Err(format!(
        "invalid vita49 endpoint: '{}' (use vita49, vita49:PORT, or vita49:IP:PORT)",
        iface
    ))
}
