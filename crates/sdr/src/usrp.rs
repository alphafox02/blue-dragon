// Copyright 2025-2026 CEMAXECUTER LLC

use crossbeam::channel::Sender;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{SampleBuf, SdrSource};

// UHD C API FFI bindings (manual, minimal)

type UhdError = c_int;
const UHD_ERROR_NONE: UhdError = 0;

// Opaque handle types
type UhdUsrpHandle = *mut c_void;
type UhdRxStreamerHandle = *mut c_void;
type UhdRxMetadataHandle = *mut c_void;
type UhdStringVectorHandle = *mut c_void;

// Tune request policy
const UHD_TUNE_REQUEST_POLICY_AUTO: c_int = 65;

// Stream modes
const UHD_STREAM_MODE_START_CONTINUOUS: c_int = 97;
const UHD_STREAM_MODE_STOP_CONTINUOUS: c_int = 111;

// RX metadata error codes
const UHD_RX_METADATA_ERROR_CODE_NONE: c_int = 0x0;
const UHD_RX_METADATA_ERROR_CODE_OVERFLOW: c_int = 0x8;

#[repr(C)]
struct UhdTuneRequest {
    target_freq: c_double,
    rf_freq_policy: c_int,
    rf_freq: c_double,
    dsp_freq_policy: c_int,
    dsp_freq: c_double,
    args: *mut c_char,
}

#[repr(C)]
struct UhdTuneResult {
    clipped_rf_freq: c_double,
    target_rf_freq: c_double,
    actual_rf_freq: c_double,
    target_dsp_freq: c_double,
    actual_dsp_freq: c_double,
}

#[repr(C)]
struct UhdStreamArgs {
    cpu_format: *mut c_char,
    otw_format: *mut c_char,
    args: *mut c_char,
    channel_list: *mut usize,
    n_channels: c_int,
}

#[repr(C)]
struct UhdStreamCmd {
    stream_mode: c_int,
    num_samps: usize,
    stream_now: bool,
    time_spec_full_secs: i64,
    time_spec_frac_secs: c_double,
}

extern "C" {
    // String vector
    fn uhd_string_vector_make(h: *mut UhdStringVectorHandle) -> UhdError;
    fn uhd_string_vector_free(h: *mut UhdStringVectorHandle) -> UhdError;
    fn uhd_string_vector_size(h: UhdStringVectorHandle, size_out: *mut usize) -> UhdError;
    fn uhd_string_vector_at(
        h: UhdStringVectorHandle,
        index: usize,
        value_out: *mut c_char,
        strbuffer_len: usize,
    ) -> UhdError;

    // USRP
    fn uhd_usrp_find(args: *const c_char, strings_out: *mut UhdStringVectorHandle) -> UhdError;
    fn uhd_usrp_make(h: *mut UhdUsrpHandle, args: *const c_char) -> UhdError;
    fn uhd_usrp_free(h: *mut UhdUsrpHandle) -> UhdError;
    fn uhd_usrp_set_rx_rate(h: UhdUsrpHandle, rate: c_double, chan: usize) -> UhdError;
    fn uhd_usrp_set_rx_gain(
        h: UhdUsrpHandle,
        gain: c_double,
        chan: usize,
        gain_name: *const c_char,
    ) -> UhdError;
    fn uhd_usrp_set_rx_freq(
        h: UhdUsrpHandle,
        tune_request: *mut UhdTuneRequest,
        chan: usize,
        tune_result: *mut UhdTuneResult,
    ) -> UhdError;
    fn uhd_usrp_set_rx_antenna(
        h: UhdUsrpHandle,
        ant: *const c_char,
        chan: usize,
    ) -> UhdError;
    fn uhd_usrp_get_rx_stream(
        h: UhdUsrpHandle,
        stream_args: *mut UhdStreamArgs,
        h_out: UhdRxStreamerHandle,
    ) -> UhdError;

    // RX Streamer
    fn uhd_rx_streamer_make(h: *mut UhdRxStreamerHandle) -> UhdError;
    fn uhd_rx_streamer_free(h: *mut UhdRxStreamerHandle) -> UhdError;
    fn uhd_rx_streamer_max_num_samps(
        h: UhdRxStreamerHandle,
        max_num_samps_out: *mut usize,
    ) -> UhdError;
    fn uhd_rx_streamer_recv(
        h: UhdRxStreamerHandle,
        buffs: *mut *mut c_void,
        samps_per_buff: usize,
        md: *mut UhdRxMetadataHandle,
        timeout: c_double,
        one_packet: bool,
        items_recvd: *mut usize,
    ) -> UhdError;
    fn uhd_rx_streamer_issue_stream_cmd(
        h: UhdRxStreamerHandle,
        stream_cmd: *const UhdStreamCmd,
    ) -> UhdError;

    // RX Metadata
    fn uhd_rx_metadata_make(handle: *mut UhdRxMetadataHandle) -> UhdError;
    fn uhd_rx_metadata_free(handle: *mut UhdRxMetadataHandle) -> UhdError;
    fn uhd_rx_metadata_error_code(
        h: UhdRxMetadataHandle,
        error_code_out: *mut c_int,
    ) -> UhdError;
}

/// Information about a detected USRP device
#[derive(Debug, Clone)]
pub struct UsrpInfo {
    pub serial: String,
    pub product: String,
    pub device_type: String,
}

/// List all available USRP devices
pub fn list_devices() -> Result<Vec<UsrpInfo>, String> {
    let args = CString::new("").unwrap();
    let mut sv: UhdStringVectorHandle = ptr::null_mut();

    unsafe {
        let err = uhd_string_vector_make(&mut sv);
        if err != UHD_ERROR_NONE {
            return Err(format!("uhd_string_vector_make failed: {}", err));
        }

        let err = uhd_usrp_find(args.as_ptr(), &mut sv);
        if err != UHD_ERROR_NONE {
            uhd_string_vector_free(&mut sv);
            return Err(format!("uhd_usrp_find failed: {}", err));
        }

        let mut count: usize = 0;
        uhd_string_vector_size(sv, &mut count);

        let mut devices = Vec::new();
        let mut buf = vec![0u8; 1024];

        for i in 0..count {
            uhd_string_vector_at(sv, i, buf.as_mut_ptr() as *mut c_char, buf.len());
            let s = std::ffi::CStr::from_ptr(buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .to_string();

            // Parse key=value pairs from the device info string
            let mut serial = String::new();
            let mut product = String::new();
            let mut device_type = String::new();

            for part in s.split(',') {
                let part = part.trim();
                if let Some((key, val)) = part.split_once('=') {
                    match key.trim() {
                        "serial" => serial = val.trim().to_string(),
                        "product" => product = val.trim().to_string(),
                        "type" => device_type = val.trim().to_string(),
                        _ => {}
                    }
                }
            }

            devices.push(UsrpInfo {
                serial,
                product,
                device_type,
            });
        }

        uhd_string_vector_free(&mut sv);
        Ok(devices)
    }
}

/// Extract serial number from interface string like "usrp-B210-SERIAL"
fn parse_serial(iface: &str) -> Option<String> {
    let parts: Vec<&str> = iface.splitn(3, '-').collect();
    if parts.len() == 3 && parts[0] == "usrp" {
        Some(parts[2].to_string())
    } else {
        None
    }
}

/// USRP SDR source using the UHD C API
pub struct UsrpSource {
    serial: String,
    sample_rate: u32,
    center_freq: u64,
    gain: f64,
    running: Arc<AtomicBool>,
}

impl UsrpSource {
    pub fn new(iface: &str, sample_rate: u32, center_freq: u64, gain: f64) -> Result<Self, String> {
        let serial = parse_serial(iface)
            .ok_or_else(|| format!("invalid USRP interface: '{}' (expected usrp-PRODUCT-SERIAL)", iface))?;

        Ok(Self {
            serial,
            sample_rate,
            center_freq,
            gain,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Get a clone of the running flag for external stop signaling
    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }
}

impl SdrSource for UsrpSource {
    fn start(&mut self, tx: Sender<SampleBuf>) -> Result<(), String> {
        self.running.store(true, Ordering::SeqCst);

        // Open USRP device
        let dev_args = CString::new(format!(
            "serial={},num_recv_frames=1024",
            self.serial
        ))
        .map_err(|e| format!("CString error: {}", e))?;

        let mut usrp: UhdUsrpHandle = ptr::null_mut();
        let empty = CString::new("").unwrap();

        unsafe {
            log::info!("opening USRP serial={}", self.serial);

            let err = uhd_usrp_make(&mut usrp, dev_args.as_ptr());
            if err != UHD_ERROR_NONE {
                return Err(format!("uhd_usrp_make failed: error {}", err));
            }

            // Set sample rate
            let err = uhd_usrp_set_rx_rate(usrp, self.sample_rate as f64, 0);
            if err != UHD_ERROR_NONE {
                uhd_usrp_free(&mut usrp);
                return Err(format!("uhd_usrp_set_rx_rate failed: error {}", err));
            }

            // Set gain
            let err = uhd_usrp_set_rx_gain(usrp, self.gain, 0, empty.as_ptr());
            if err != UHD_ERROR_NONE {
                uhd_usrp_free(&mut usrp);
                return Err(format!("uhd_usrp_set_rx_gain failed: error {}", err));
            }

            // Tune to center frequency
            let mut tune_req = UhdTuneRequest {
                target_freq: self.center_freq as f64,
                rf_freq_policy: UHD_TUNE_REQUEST_POLICY_AUTO,
                rf_freq: 0.0,
                dsp_freq_policy: UHD_TUNE_REQUEST_POLICY_AUTO,
                dsp_freq: 0.0,
                args: ptr::null_mut(),
            };
            let mut tune_result = UhdTuneResult {
                clipped_rf_freq: 0.0,
                target_rf_freq: 0.0,
                actual_rf_freq: 0.0,
                target_dsp_freq: 0.0,
                actual_dsp_freq: 0.0,
            };
            let err = uhd_usrp_set_rx_freq(usrp, &mut tune_req, 0, &mut tune_result);
            if err != UHD_ERROR_NONE {
                uhd_usrp_free(&mut usrp);
                return Err(format!("uhd_usrp_set_rx_freq failed: error {}", err));
            }

            log::info!(
                "USRP tuned: RF={:.1} MHz, DSP={:.1} kHz",
                tune_result.actual_rf_freq / 1e6,
                tune_result.actual_dsp_freq / 1e3,
            );

            // Create RX streamer
            let mut rx_handle: UhdRxStreamerHandle = ptr::null_mut();
            let err = uhd_rx_streamer_make(&mut rx_handle);
            if err != UHD_ERROR_NONE {
                uhd_usrp_free(&mut usrp);
                return Err(format!("uhd_rx_streamer_make failed: error {}", err));
            }

            // Create RX metadata handle
            let mut md: UhdRxMetadataHandle = ptr::null_mut();
            let err = uhd_rx_metadata_make(&mut md);
            if err != UHD_ERROR_NONE {
                uhd_rx_streamer_free(&mut rx_handle);
                uhd_usrp_free(&mut usrp);
                return Err(format!("uhd_rx_metadata_make failed: error {}", err));
            }

            // Configure stream (SC16 format for full 12-bit ADC precision)
            let cpu_fmt = CString::new("sc16").unwrap();
            let otw_fmt = CString::new("sc16").unwrap();
            let stream_args_str = CString::new("").unwrap();
            let mut channel: usize = 0;

            let mut stream_args = UhdStreamArgs {
                cpu_format: cpu_fmt.as_ptr() as *mut c_char,
                otw_format: otw_fmt.as_ptr() as *mut c_char,
                args: stream_args_str.as_ptr() as *mut c_char,
                channel_list: &mut channel,
                n_channels: 1,
            };

            let err = uhd_usrp_get_rx_stream(usrp, &mut stream_args, rx_handle);
            if err != UHD_ERROR_NONE {
                uhd_rx_metadata_free(&mut md);
                uhd_rx_streamer_free(&mut rx_handle);
                uhd_usrp_free(&mut usrp);
                return Err(format!("uhd_usrp_get_rx_stream failed: error {}", err));
            }

            // Get max samples per recv call
            let mut max_samps: usize = 0;
            uhd_rx_streamer_max_num_samps(rx_handle, &mut max_samps);
            log::info!("USRP max_num_samps: {}", max_samps);

            // Start continuous streaming
            let stream_cmd = UhdStreamCmd {
                stream_mode: UHD_STREAM_MODE_START_CONTINUOUS,
                num_samps: 0,
                stream_now: true,
                time_spec_full_secs: 0,
                time_spec_frac_secs: 0.0,
            };
            let err = uhd_rx_streamer_issue_stream_cmd(rx_handle, &stream_cmd);
            if err != UHD_ERROR_NONE {
                uhd_rx_metadata_free(&mut md);
                uhd_rx_streamer_free(&mut rx_handle);
                uhd_usrp_free(&mut usrp);
                return Err(format!("uhd_rx_streamer_issue_stream_cmd failed: error {}", err));
            }

            log::info!("USRP streaming started ({} MHz, {} MS/s, gain={} dB)",
                self.center_freq / 1_000_000,
                self.sample_rate / 1_000_000,
                self.gain,
            );

            // Pre-allocate receive buffer (SC16: 2 bytes per component, 4 bytes per complex sample)
            let mut sc16_buf = vec![0i16; max_samps * 2];
            let mut overflow_count: u64 = 0;

            // Main receive loop
            while self.running.load(Ordering::SeqCst) {
                let mut buf_ptr = sc16_buf.as_mut_ptr() as *mut c_void;
                let mut num_rx: usize = 0;

                let err = uhd_rx_streamer_recv(
                    rx_handle,
                    &mut buf_ptr,
                    max_samps,
                    &mut md,
                    3.0,   // timeout seconds
                    false, // one_packet
                    &mut num_rx,
                );

                if err != UHD_ERROR_NONE {
                    log::error!("uhd_rx_streamer_recv error: {}", err);
                    break;
                }

                // Check metadata error code
                let mut error_code: c_int = 0;
                uhd_rx_metadata_error_code(md, &mut error_code);

                if error_code != UHD_RX_METADATA_ERROR_CODE_NONE
                    && error_code != UHD_RX_METADATA_ERROR_CODE_OVERFLOW
                {
                    log::error!("USRP streaming error: {}", error_code);
                    break;
                }

                if error_code == UHD_RX_METADATA_ERROR_CODE_OVERFLOW {
                    overflow_count += 1;
                }

                if num_rx == 0 {
                    continue;
                }

                // SC16 samples are already i16 -- copy directly
                let data = sc16_buf[..num_rx * 2].to_vec();

                if tx.send(SampleBuf { data, num_samples: num_rx }).is_err() {
                    break; // receiver dropped
                }
            }

            // Stop streaming
            let stop_cmd = UhdStreamCmd {
                stream_mode: UHD_STREAM_MODE_STOP_CONTINUOUS,
                num_samps: 0,
                stream_now: true,
                time_spec_full_secs: 0,
                time_spec_frac_secs: 0.0,
            };
            let _ = uhd_rx_streamer_issue_stream_cmd(rx_handle, &stop_cmd);

            // Cleanup
            uhd_rx_metadata_free(&mut md);
            uhd_rx_streamer_free(&mut rx_handle);
            uhd_usrp_free(&mut usrp);

            if overflow_count > 0 {
                log::warn!("USRP: {} overflows during capture", overflow_count);
            }
            log::info!("USRP streaming stopped");
        }

        Ok(())
    }

    fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn center_frequency(&self) -> u64 {
        self.center_freq
    }
}

/// Opened USRP device handle for direct recv_into() calls.
/// Uses SC16 wire format for full 12-bit ADC precision.
pub struct UsrpHandle {
    usrp: UhdUsrpHandle,
    rx_handle: UhdRxStreamerHandle,
    md: UhdRxMetadataHandle,
    max_samps: usize,
    pub running: Arc<AtomicBool>,
    overflow_count: u64,
}

impl UsrpHandle {
    /// Open USRP and start streaming. Returns a handle for recv_into() calls.
    /// `antenna`: optional RX port name (e.g. "RX2", "TX/RX"). Defaults to UHD default (RX2).
    pub fn open(iface: &str, sample_rate: u32, center_freq: u64, gain: f64, antenna: Option<&str>) -> Result<Self, String> {
        let serial = parse_serial(iface)
            .ok_or_else(|| format!("invalid USRP interface: '{}'", iface))?;

        let dev_args = CString::new(format!(
            "serial={},num_recv_frames=1024",
            serial
        ))
        .map_err(|e| format!("CString error: {}", e))?;
        let empty = CString::new("").unwrap();

        unsafe {
            let mut usrp: UhdUsrpHandle = ptr::null_mut();
            let err = uhd_usrp_make(&mut usrp, dev_args.as_ptr());
            if err != UHD_ERROR_NONE {
                return Err(format!("uhd_usrp_make failed: error {}", err));
            }

            uhd_usrp_set_rx_rate(usrp, sample_rate as f64, 0);
            uhd_usrp_set_rx_gain(usrp, gain, 0, empty.as_ptr());

            if let Some(ant) = antenna {
                let ant_c = CString::new(ant)
                    .map_err(|e| format!("invalid antenna name: {}", e))?;
                let err = uhd_usrp_set_rx_antenna(usrp, ant_c.as_ptr(), 0);
                if err != UHD_ERROR_NONE {
                    uhd_usrp_free(&mut usrp);
                    return Err(format!("uhd_usrp_set_rx_antenna('{}') failed: error {}", ant, err));
                }
                log::info!("USRP antenna set to '{}'", ant);
            }

            let mut tune_req = UhdTuneRequest {
                target_freq: center_freq as f64,
                rf_freq_policy: UHD_TUNE_REQUEST_POLICY_AUTO,
                rf_freq: 0.0,
                dsp_freq_policy: UHD_TUNE_REQUEST_POLICY_AUTO,
                dsp_freq: 0.0,
                args: ptr::null_mut(),
            };
            let mut tune_result = UhdTuneResult {
                clipped_rf_freq: 0.0,
                target_rf_freq: 0.0,
                actual_rf_freq: 0.0,
                target_dsp_freq: 0.0,
                actual_dsp_freq: 0.0,
            };
            uhd_usrp_set_rx_freq(usrp, &mut tune_req, 0, &mut tune_result);

            let mut rx_handle: UhdRxStreamerHandle = ptr::null_mut();
            uhd_rx_streamer_make(&mut rx_handle);

            let mut md: UhdRxMetadataHandle = ptr::null_mut();
            uhd_rx_metadata_make(&mut md);

            let cpu_fmt = CString::new("sc16").unwrap();
            let otw_fmt = CString::new("sc16").unwrap();
            let stream_args_str = CString::new("").unwrap();
            let mut channel: usize = 0;

            let mut stream_args = UhdStreamArgs {
                cpu_format: cpu_fmt.as_ptr() as *mut c_char,
                otw_format: otw_fmt.as_ptr() as *mut c_char,
                args: stream_args_str.as_ptr() as *mut c_char,
                channel_list: &mut channel,
                n_channels: 1,
            };

            uhd_usrp_get_rx_stream(usrp, &mut stream_args, rx_handle);

            let mut max_samps: usize = 0;
            uhd_rx_streamer_max_num_samps(rx_handle, &mut max_samps);

            let stream_cmd = UhdStreamCmd {
                stream_mode: UHD_STREAM_MODE_START_CONTINUOUS,
                num_samps: 0,
                stream_now: true,
                time_spec_full_secs: 0,
                time_spec_frac_secs: 0.0,
            };
            uhd_rx_streamer_issue_stream_cmd(rx_handle, &stream_cmd);

            let running = Arc::new(AtomicBool::new(true));

            Ok(Self {
                usrp,
                rx_handle,
                md,
                max_samps,
                running,
                overflow_count: 0,
            })
        }
    }

    /// Receive SC16 samples directly into an i16 buffer.
    /// `buf`: destination buffer (i16 IQ pairs, at least max_samps * 2 elements).
    /// Returns: number of complex samples received, or 0 on timeout/error.
    pub fn recv_into_i16(&mut self, buf: &mut [i16]) -> usize {
        let max_samps = (buf.len() / 2).min(self.max_samps);
        let mut num_rx: usize = 0;

        unsafe {
            let mut buf_ptr = buf.as_mut_ptr() as *mut c_void;

            let err = uhd_rx_streamer_recv(
                self.rx_handle,
                &mut buf_ptr,
                max_samps,
                &mut self.md,
                3.0,
                false,
                &mut num_rx,
            );

            if err != UHD_ERROR_NONE {
                return 0;
            }

            let mut error_code: c_int = 0;
            uhd_rx_metadata_error_code(self.md, &mut error_code);

            if error_code == UHD_RX_METADATA_ERROR_CODE_OVERFLOW {
                self.overflow_count += 1;
            } else if error_code != UHD_RX_METADATA_ERROR_CODE_NONE {
                return 0;
            }
        }

        num_rx
    }

    /// Set RX gain at runtime (thread-safe FFI call).
    pub fn set_gain(&self, gain: f64) {
        let empty = CString::new("").unwrap();
        unsafe {
            uhd_usrp_set_rx_gain(self.usrp, gain, 0, empty.as_ptr());
        }
    }

    pub fn max_samps(&self) -> usize {
        self.max_samps
    }

    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }
}

impl Drop for UsrpHandle {
    fn drop(&mut self) {
        unsafe {
            let stop_cmd = UhdStreamCmd {
                stream_mode: UHD_STREAM_MODE_STOP_CONTINUOUS,
                num_samps: 0,
                stream_now: true,
                time_spec_full_secs: 0,
                time_spec_frac_secs: 0.0,
            };
            let _ = uhd_rx_streamer_issue_stream_cmd(self.rx_handle, &stop_cmd);

            uhd_rx_metadata_free(&mut self.md);
            uhd_rx_streamer_free(&mut self.rx_handle);
            uhd_usrp_free(&mut self.usrp);
        }

        if self.overflow_count > 0 {
            eprintln!("USRP: {} overflows during capture", self.overflow_count);
        }
    }
}
