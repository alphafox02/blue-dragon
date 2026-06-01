// Copyright 2025-2026 CEMAXECUTER LLC
//
// Epiq Solutions Sidekiq backend (AD9361-based: Stretch, m.2, Z2, Z3u, X40, ...).
// Mirrors the SDR++ sidekiq_source module in C, but exposes the blue-dragon
// SdrSource + open()/recv_into_i16() handle surface used elsewhere in this crate.

use crossbeam::channel::Sender;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{SampleBuf, SdrSource};

// ---------------------------------------------------------------------------
// libsidekiq enum mirrors (verbatim values from sidekiq_xport_types.h /
// sidekiq_api_types.h / sidekiq_types.h in SDK v4.25.0).
// ---------------------------------------------------------------------------

// skiq_xport_type_t
const SKIQ_XPORT_TYPE_PCIE: c_int = 0;
#[allow(dead_code)]
const SKIQ_XPORT_TYPE_USB: c_int = 1;
#[allow(dead_code)]
const SKIQ_XPORT_TYPE_CUSTOM: c_int = 2;
#[allow(dead_code)]
const SKIQ_XPORT_TYPE_NET: c_int = 3;
#[allow(dead_code)]
const SKIQ_XPORT_TYPE_MAX: c_int = 4;
const SKIQ_XPORT_TYPE_AUTO: c_int = 5;

// skiq_xport_init_level_t
#[allow(dead_code)]
const SKIQ_XPORT_INIT_LEVEL_BASIC: c_int = 0;
const SKIQ_XPORT_INIT_LEVEL_FULL: c_int = 1;

// skiq_rx_hdl_t
const SKIQ_RX_HDL_A1: c_int = 0;
const SKIQ_RX_HDL_A2: c_int = 1;
const SKIQ_RX_HDL_B1: c_int = 2;
const SKIQ_RX_HDL_B2: c_int = 3;

// skiq_rf_port_t
const SKIQ_RF_PORT_J1: c_int = 0;
const SKIQ_RF_PORT_J2: c_int = 1;

// skiq_rx_gain_t
const SKIQ_RX_GAIN_MANUAL: c_int = 0;
#[allow(dead_code)]
const SKIQ_RX_GAIN_AUTO: c_int = 1;

// skiq_iq_order_t
const SKIQ_IQ_ORDER_QI: c_int = 0;
#[allow(dead_code)]
const SKIQ_IQ_ORDER_IQ: c_int = 1;

// skiq_rx_stream_mode_t
const SKIQ_RX_STREAM_MODE_HIGH_TPUT: c_int = 0;
#[allow(dead_code)]
const SKIQ_RX_STREAM_MODE_LOW_LATENCY: c_int = 1;
#[allow(dead_code)]
const SKIQ_RX_STREAM_MODE_BALANCED: c_int = 2;

// skiq_rx_status_t
const SKIQ_RX_STATUS_SUCCESS: c_int = 0;
const SKIQ_RX_STATUS_NO_DATA: c_int = -1;

const SKIQ_MAX_NUM_CARDS: usize = 32;

/// skiq_rx_block_t header is 24 bytes (3 * uint64_t) before the int16_t data[]
/// flexible array, per sidekiq_types.h SKIQ_RX_HEADER_SIZE_IN_BYTES.
const SKIQ_RX_HEADER_SIZE_IN_BYTES: u32 = 24;

// Link directives come from build.rs (the static archive's name carries an
// arch/SDK-target suffix, e.g. libsidekiq__x86_64.gcc.a).
extern "C" {
    fn skiq_init_without_cards() -> c_int;
    fn skiq_exit() -> c_int;
    fn skiq_get_cards(xport_type: c_int, p_num_cards: *mut u8, p_cards: *mut u8) -> c_int;
    fn skiq_enable_cards(cards: *const u8, num_cards: u8, level: c_int) -> c_int;
    fn skiq_disable_cards(cards: *const u8, num_cards: u8) -> c_int;
    fn skiq_read_serial_string(card: u8, pp_serial_num: *mut *mut c_char) -> c_int;
    fn skiq_read_rx_gain_index_range(
        card: u8,
        hdl: c_int,
        p_min: *mut u8,
        p_max: *mut u8,
    ) -> c_int;
    fn skiq_write_iq_order_mode(card: u8, mode: c_int) -> c_int;
    fn skiq_write_rx_sample_rate_and_bandwidth(
        card: u8,
        hdl: c_int,
        rate: u32,
        bandwidth: u32,
    ) -> c_int;
    fn skiq_write_rx_LO_freq(card: u8, hdl: c_int, freq: u64) -> c_int;
    fn skiq_write_rx_gain_mode(card: u8, hdl: c_int, mode: c_int) -> c_int;
    fn skiq_write_rx_gain(card: u8, hdl: c_int, gain_index: u8) -> c_int;
    fn skiq_write_rx_stream_mode(card: u8, mode: c_int) -> c_int;
    fn skiq_set_rx_transfer_timeout(card: u8, timeout_us: i32) -> c_int;
    fn skiq_start_rx_streaming(card: u8, hdl: c_int) -> c_int;
    fn skiq_stop_rx_streaming(card: u8, hdl: c_int) -> c_int;
    fn skiq_receive(
        card: u8,
        p_hdl: *mut c_int,
        pp_block: *mut *mut c_void,
        p_data_len: *mut u32,
    ) -> c_int;
    fn skiq_read_rx_block_size(card: u8, stream_mode: c_int) -> i32;
    fn skiq_write_rx_rf_port_for_hdl(card: u8, hdl: c_int, port: c_int) -> c_int;
    fn skiq_read_rx_rf_port_for_hdl(card: u8, hdl: c_int, p_port: *mut c_int) -> c_int;
    fn skiq_read_rx_sample_rate_and_bandwidth(
        card: u8,
        hdl: c_int,
        p_rate: *mut u32,
        p_actual_rate: *mut f64,
        p_bandwidth: *mut u32,
        p_actual_bandwidth: *mut u32,
    ) -> c_int;
    fn skiq_write_rx_dc_offset_corr(card: u8, hdl: c_int, enable: bool) -> c_int;
    fn skiq_read_rx_iq_resolution(card: u8, p_adc_res: *mut u8) -> c_int;
    fn skiq_read_rx_LO_freq_range(card: u8, p_max: *mut u64, p_min: *mut u64) -> c_int;
    fn skiq_read_part_info(
        card: u8,
        p_part_number: *mut c_char,
        p_revision: *mut c_char,
        p_variant: *mut c_char,
    ) -> c_int;
}

// Buffer sizes for skiq_read_part_info (from sidekiq_types.h).
const SKIQ_PART_NUM_STRLEN: usize = 7;
const SKIQ_REVISION_STRLEN: usize = 3;
const SKIQ_VARIANT_STRLEN: usize = 16;

// ---------------------------------------------------------------------------
// Process-wide SDK lifetime guard. libsidekiq is a singleton; calling
// skiq_init_without_cards() twice without an intervening skiq_exit() returns
// an error. Use a refcount so list_devices() + an open SidekiqHandle can
// coexist without either tearing the SDK down underneath the other.
// ---------------------------------------------------------------------------

use std::sync::{Mutex, OnceLock};

struct SdkGuard {
    refcount: u32,
}

fn sdk_state() -> &'static Mutex<SdkGuard> {
    static STATE: OnceLock<Mutex<SdkGuard>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(SdkGuard { refcount: 0 }))
}

fn sdk_acquire() -> Result<(), String> {
    let mut g = sdk_state().lock().map_err(|_| "sidekiq SDK mutex poisoned")?;
    if g.refcount == 0 {
        let rc = unsafe { skiq_init_without_cards() };
        if rc != 0 {
            return Err(format!("skiq_init_without_cards failed: {}", rc));
        }
    }
    g.refcount += 1;
    Ok(())
}

fn sdk_release() {
    if let Ok(mut g) = sdk_state().lock() {
        if g.refcount > 0 {
            g.refcount -= 1;
            if g.refcount == 0 {
                unsafe { skiq_exit() };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Enumeration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SidekiqInfo {
    /// Card id assigned by libsidekiq for this enumeration session.
    pub card: u8,
    /// Serial number reported by the device (stable across reboots).
    pub serial: String,
}

pub fn list_devices() -> Result<Vec<SidekiqInfo>, String> {
    sdk_acquire()?;

    let mut cards = [0u8; SKIQ_MAX_NUM_CARDS];
    let mut num: u8 = 0;
    let rc = unsafe { skiq_get_cards(SKIQ_XPORT_TYPE_AUTO, &mut num, cards.as_mut_ptr()) };
    if rc != 0 {
        sdk_release();
        return Err(format!("skiq_get_cards failed: {}", rc));
    }

    let mut out = Vec::with_capacity(num as usize);
    for i in 0..num as usize {
        let card = cards[i];
        // Bring card up at BASIC level just long enough to read the serial.
        let r = unsafe { skiq_enable_cards(&card, 1, SKIQ_XPORT_INIT_LEVEL_BASIC) };
        if r != 0 {
            // Skip ghost ids that won't enable.
            continue;
        }
        let mut p_serial: *mut c_char = ptr::null_mut();
        let serial = if unsafe { skiq_read_serial_string(card, &mut p_serial) } == 0
            && !p_serial.is_null()
        {
            unsafe { std::ffi::CStr::from_ptr(p_serial) }
                .to_string_lossy()
                .into_owned()
        } else {
            String::new()
        };
        let _ = unsafe { skiq_disable_cards(&card, 1) };

        out.push(SidekiqInfo { card, serial });
    }

    sdk_release();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Interface / antenna parsing.
//
// Interface string: "sidekiq-<serial>". A bare "sidekiq" picks the first card.
// Antenna string: "<handle>" or "<handle>:<port>", where:
//   handle  ∈ {A1, A2, B1, B2}  (default A1)
//   port    ∈ {J1, J2}          (default: leave whatever the SDK picks)
// ---------------------------------------------------------------------------

fn parse_serial(iface: &str) -> Option<String> {
    if let Some(rest) = iface.strip_prefix("sidekiq-") {
        if rest.is_empty() {
            None
        } else {
            Some(rest.to_string())
        }
    } else if iface == "sidekiq" {
        None
    } else {
        None
    }
}

fn parse_antenna(antenna: Option<&str>) -> (c_int, Option<c_int>) {
    let s = match antenna {
        Some(s) => s.to_uppercase(),
        None => return (SKIQ_RX_HDL_A1, None),
    };
    let (hdl_s, port_s) = match s.split_once(':') {
        Some((a, b)) => (a, Some(b)),
        None => (s.as_str(), None),
    };
    let hdl = match hdl_s {
        "A1" => SKIQ_RX_HDL_A1,
        "A2" => SKIQ_RX_HDL_A2,
        "B1" => SKIQ_RX_HDL_B1,
        "B2" => SKIQ_RX_HDL_B2,
        _ => SKIQ_RX_HDL_A1,
    };
    let port = port_s.and_then(|p| match p {
        "J1" => Some(SKIQ_RF_PORT_J1),
        "J2" => Some(SKIQ_RF_PORT_J2),
        _ => None,
    });
    (hdl, port)
}

/// Locate the chosen card. If a serial was provided we search for it; else we
/// take the first valid card returned by the SDK.
fn pick_card(serial_filter: Option<&str>) -> Result<u8, String> {
    let mut cards = [0u8; SKIQ_MAX_NUM_CARDS];
    let mut num: u8 = 0;
    let rc = unsafe { skiq_get_cards(SKIQ_XPORT_TYPE_AUTO, &mut num, cards.as_mut_ptr()) };
    if rc != 0 {
        return Err(format!("skiq_get_cards failed: {}", rc));
    }
    if num == 0 {
        return Err("no Sidekiq cards present".into());
    }

    if let Some(want) = serial_filter {
        for i in 0..num as usize {
            let card = cards[i];
            // Probe serial via BASIC enable, then drop it.
            if unsafe { skiq_enable_cards(&card, 1, SKIQ_XPORT_INIT_LEVEL_BASIC) } != 0 {
                continue;
            }
            let mut p_serial: *mut c_char = ptr::null_mut();
            let got = if unsafe { skiq_read_serial_string(card, &mut p_serial) } == 0
                && !p_serial.is_null()
            {
                unsafe { std::ffi::CStr::from_ptr(p_serial) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };
            let _ = unsafe { skiq_disable_cards(&card, 1) };
            if got.eq_ignore_ascii_case(want) {
                return Ok(card);
            }
        }
        Err(format!("Sidekiq serial '{}' not found", want))
    } else {
        Ok(cards[0])
    }
}

// ---------------------------------------------------------------------------
// Zero-copy handle (matches the recv_into_i16 path used by pipeline.rs).
// ---------------------------------------------------------------------------

pub struct SidekiqHandle {
    card: u8,
    hdl: c_int,
    /// Target samples delivered per recv_into_i16 call. Multiple SDK blocks
    /// are aggregated per call to deliver longer contiguous chunks (~2 ms at
    /// the configured rate) so the downstream PFB / burst detector sees the
    /// same chunk sizes as bladeRF / USRP backends.
    max_samps: usize,
    /// Set true once skiq_enable_cards(FULL) succeeded so Drop knows to
    /// disable + skiq_exit.
    enabled: bool,
    /// Set true once skiq_start_rx_streaming succeeded.
    streaming: bool,
    /// Cumulative count of RFIC overload indications (overload bit in the
    /// block header).
    overflow_count: u64,
    /// Cumulative count of detected sample drops (rf_timestamp gaps between
    /// consecutive blocks). Distinct from overflow_count, this signals USB /
    /// queue starvation, not RF saturation.
    drop_count: u64,
    /// Left-shift applied to each 16-bit sample to fill the i16 numeric
    /// range. Derived from the device-reported IQ resolution:
    ///   - 12-bit ADC (AD9361/9364, Stretch/M.2/Z2/Z3u/Z4): shift = 4
    ///   - 14-bit ADC: shift = 2
    ///   - 16-bit ADC (ADRV9004/9009, Nv100/Nvm2/X2/X4/X40): shift = 0
    iq_shift: u32,
    /// Last seen rf_timestamp + expected next-block rf_timestamp, for gap
    /// detection. None until the first block is received.
    last_ts_next: Option<u64>,
    pub running: Arc<AtomicBool>,
}

unsafe impl Send for SidekiqHandle {}

/// Optional tuning knobs surfaced as separate args so we don't break the
/// existing open() signature used by pipeline.rs. None / false / default-everything
/// reproduces the conservative behaviour.
#[derive(Debug, Clone, Default)]
pub struct SidekiqExtras {
    /// Switch the RFIC's built-in AGC on. When false (default) we use a
    /// manual gain index from the `gain` argument; the actual index range
    /// is read from the device at open and clamped accordingly.
    pub agc: bool,
    /// Enable FPGA DC offset correction (very near-DC 1-pole high-pass). Most
    /// AD9361-based SKUs support this. No-op + warning if the SKU does not.
    pub dc_offset_corr: bool,
    /// Best-effort: lower the process nice value so recv stays responsive
    /// when the host is loaded. Silently no-ops without CAP_SYS_NICE.
    pub boost_priority: bool,
}

impl SidekiqHandle {
    pub fn open(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: f64,
        antenna: Option<&str>,
    ) -> Result<Self, String> {
        // Defaults: enable DC correction + preselect targeting (low risk on
        // supported SKUs, harmless warnings elsewhere). AGC stays off so the
        // -g index argument still controls gain by default.
        Self::open_with_extras(
            iface,
            sample_rate,
            center_freq,
            gain,
            antenna,
            SidekiqExtras {
                agc: false,
                dc_offset_corr: true,
                boost_priority: true,
            },
        )
    }

    pub fn open_with_extras(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: f64,
        antenna: Option<&str>,
        extras: SidekiqExtras,
    ) -> Result<Self, String> {
        if extras.boost_priority {
            boost_recv_thread();
        }
        sdk_acquire()?;

        let want_serial = parse_serial(iface);
        let card = match pick_card(want_serial.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                sdk_release();
                return Err(e);
            }
        };

        let (hdl, port) = parse_antenna(antenna);

        // FULL enable (after a defensive disable to clear any prior state).
        let _ = unsafe { skiq_disable_cards(&card, 1) };
        let r = unsafe { skiq_enable_cards(&card, 1, SKIQ_XPORT_INIT_LEVEL_FULL) };
        if r != 0 {
            sdk_release();
            return Err(format!("skiq_enable_cards(FULL) failed: {}", r));
        }

        // ---- Query device-reported metadata so we adapt per-product instead
        // of hardcoding limits. The Sidekiq family spans multiple RFICs:
        // AD9361/9364 (12-bit), ADRV9004 (16-bit), ADRV9009 (16-bit), etc.,
        // each with its own sample-rate range, LO range, and gain index range.
        let mut part_num = [0i8; SKIQ_PART_NUM_STRLEN];
        let mut part_rev = [0i8; SKIQ_REVISION_STRLEN];
        let mut part_var = [0i8; SKIQ_VARIANT_STRLEN];
        let part_str = if unsafe {
            skiq_read_part_info(
                card,
                part_num.as_mut_ptr() as *mut c_char,
                part_rev.as_mut_ptr() as *mut c_char,
                part_var.as_mut_ptr() as *mut c_char,
            )
        } == 0
        {
            let n = unsafe { std::ffi::CStr::from_ptr(part_num.as_ptr() as *const c_char) }
                .to_string_lossy()
                .into_owned();
            let r = unsafe { std::ffi::CStr::from_ptr(part_rev.as_ptr() as *const c_char) }
                .to_string_lossy()
                .into_owned();
            let v = unsafe { std::ffi::CStr::from_ptr(part_var.as_ptr() as *const c_char) }
                .to_string_lossy()
                .into_owned();
            format!("{} rev {} variant {}", n, r, v)
        } else {
            "unknown".to_string()
        };

        // Validate LO is in the device's tunable range.
        let mut lo_max: u64 = 0;
        let mut lo_min: u64 = 0;
        if unsafe { skiq_read_rx_LO_freq_range(card, &mut lo_max, &mut lo_min) } == 0
            && lo_max >= lo_min
            && (center_freq < lo_min || center_freq > lo_max)
        {
            unsafe { skiq_disable_cards(&card, 1) };
            sdk_release();
            return Err(format!(
                "sidekiq: requested LO {} Hz is outside this device's RX tuning range \
                 ({} Hz .. {} Hz). Pick a different -c.",
                center_freq, lo_min, lo_max
            ));
        }

        // ADC resolution → determines the i16 scaling shift in recv_into_i16.
        // Falls back to 12-bit (shift 4) if the SDK does not implement this
        // call on a given device, which matches the legacy behaviour.
        let mut adc_bits: u8 = 12;
        let _ = unsafe { skiq_read_rx_iq_resolution(card, &mut adc_bits) };
        let iq_shift = (16u32).saturating_sub(adc_bits as u32);

        log::info!(
            "sidekiq device metadata: part={}, RX ADC bits={}, LO range {}..{} Hz",
            part_str, adc_bits, lo_min, lo_max
        );

        // Helper to roll back partial setup on any subsequent failure.
        let teardown = |card: u8, streaming: bool| {
            if streaming {
                unsafe { skiq_stop_rx_streaming(card, SKIQ_RX_HDL_A1) };
            }
            unsafe { skiq_disable_cards(&card, 1) };
            sdk_release();
        };

        // Q,I native order (matches the SDR++ module; we'll swap to I,Q in recv).
        let _ = unsafe { skiq_write_iq_order_mode(card, SKIQ_IQ_ORDER_QI) };

        // RF port: explicit override > default J1. On the m.2-2280 (and other
        // dual-port SKUs) J1 is the RX1-labeled port and J2 is the TX/RX2
        // duplexed port; both can be used for RX but the optimal choice depends
        // on hardware variant and how the user has cabled the antennas. Some
        // SKUs only expose one valid port for a given handle, in which case
        // the write is a no-op; we read back and log what the device reports.
        let want_port = port.unwrap_or(SKIQ_RF_PORT_J1);
        let _ = unsafe { skiq_write_rx_rf_port_for_hdl(card, hdl, want_port) };
        let mut actual_port: c_int = -1;
        let port_name = if unsafe {
            skiq_read_rx_rf_port_for_hdl(card, hdl, &mut actual_port)
        } == 0
        {
            port_name(actual_port)
        } else {
            "?"
        };

        // Sample rate + bandwidth. We request bandwidth == sample_rate and
        // let the SDK round to the nearest supported analog channel filter
        // for this hardware; the readback below logs the actual values the
        // device negotiated.
        let bandwidth = sample_rate;
        let r = unsafe {
            skiq_write_rx_sample_rate_and_bandwidth(card, hdl, sample_rate, bandwidth)
        };
        if r != 0 {
            log::warn!(
                "sidekiq: write_rx_sample_rate_and_bandwidth({}, {}) failed rc={} \
                 (device may require a fixed profile, continuing)",
                sample_rate, bandwidth, r
            );
        }

        // Verify the configured rate matches what was requested. blue-dragon's
        // pipeline assumes sample_rate == channels * 1_000_000; if the device
        // negotiated a different rate (any radio's clock generator has discrete
        // valid rates), symbol timing for BLE GFSK would drift across each
        // packet. We surface a clear error so the user can pick a -C value
        // that lines up with this device's supported rates.
        let mut got_rate: u32 = 0;
        let mut got_actual: f64 = 0.0;
        let mut got_bw: u32 = 0;
        let mut got_actual_bw: u32 = 0;
        let rc_rb = unsafe {
            skiq_read_rx_sample_rate_and_bandwidth(
                card,
                hdl,
                &mut got_rate,
                &mut got_actual,
                &mut got_bw,
                &mut got_actual_bw,
            )
        };
        if rc_rb == 0 {
            let drift = (got_actual - sample_rate as f64) / sample_rate as f64;
            log::info!(
                "sidekiq: requested {} Sps, configured {} Sps, actual {:.3} Sps \
                 (drift {:+.4}%); BW req {} actual {}",
                sample_rate, got_rate, got_actual, drift * 100.0, bandwidth, got_actual_bw
            );
            if drift.abs() > 0.0005 {
                teardown(card, false);
                return Err(format!(
                    "sidekiq: requested {} Sps but device is running at {:.0} Sps \
                     (drift {:+.3}%). blue-dragon assumes sample_rate == channels \
                     * 1_000_000 exactly; pick a -C value whose Hz rate is in \
                     this device's supported set (e.g. -C 40, -C 20, -C 10).",
                    sample_rate, got_actual, drift * 100.0
                ));
            }
        }

        let r = unsafe { skiq_write_rx_LO_freq(card, hdl, center_freq) };
        if r != 0 {
            teardown(card, false);
            return Err(format!("skiq_write_rx_LO_freq({}) failed: {}", center_freq, r));
        }

        // Gain: AGC if explicitly requested, otherwise manual index. blue-dragon
        // passes gain as f64 dB but the Sidekiq RX gain knob is a 0..N index
        // (typical max 76 on AD9361 parts); we use it as the index directly.
        let mut gmin: u8 = 0;
        let mut gmax: u8 = 76;
        let _ = unsafe { skiq_read_rx_gain_index_range(card, hdl, &mut gmin, &mut gmax) };
        let gi = (gain.round() as i32).clamp(gmin as i32, gmax as i32) as u8;
        if extras.agc {
            let r = unsafe { skiq_write_rx_gain_mode(card, hdl, SKIQ_RX_GAIN_AUTO) };
            if r != 0 {
                log::warn!(
                    "sidekiq: write_rx_gain_mode(auto) failed rc={}, falling back to manual idx {}",
                    r, gi
                );
                let _ = unsafe { skiq_write_rx_gain_mode(card, hdl, SKIQ_RX_GAIN_MANUAL) };
                let _ = unsafe { skiq_write_rx_gain(card, hdl, gi) };
            } else {
                log::info!("sidekiq: gain mode = AGC (auto)");
            }
        } else {
            let _ = unsafe { skiq_write_rx_gain_mode(card, hdl, SKIQ_RX_GAIN_MANUAL) };
            let r = unsafe { skiq_write_rx_gain(card, hdl, gi) };
            if r != 0 {
                log::warn!("sidekiq: write_rx_gain({}) failed: {}", gi, r);
            }
        }

        // Optional: enable FPGA DC offset correction. Removes residual LO
        // leakage that otherwise lands at the center of the captured band and
        // can systematically corrupt bits on near-DC channels.
        if extras.dc_offset_corr {
            let r = unsafe { skiq_write_rx_dc_offset_corr(card, hdl, true) };
            if r == 0 {
                log::info!("sidekiq: FPGA DC offset correction enabled");
            } else {
                log::info!(
                    "sidekiq: DC offset correction not available on this SKU (rc={}), continuing",
                    r
                );
            }
        }

        // Preselect filter is left to the SDK's auto-selection. Per the SDK
        // docs, the filter encompassing the configured LO frequency is
        // automatically applied when skiq_write_rx_LO_freq is called.

        let _ = unsafe { skiq_write_rx_stream_mode(card, SKIQ_RX_STREAM_MODE_HIGH_TPUT) };
        let _ = unsafe { skiq_set_rx_transfer_timeout(card, 50) };

        let block_bytes = unsafe { skiq_read_rx_block_size(card, SKIQ_RX_STREAM_MODE_HIGH_TPUT) };
        if block_bytes <= SKIQ_RX_HEADER_SIZE_IN_BYTES as i32 {
            teardown(card, false);
            return Err(format!("invalid block size {}", block_bytes));
        }
        let block_samps =
            ((block_bytes as u32 - SKIQ_RX_HEADER_SIZE_IN_BYTES) / 4) as usize;
        // Target ~2 ms of audio per recv_into_i16 call (similar order to what
        // bladeRF and USRP deliver), aggregating multiple SDK blocks. At
        // 40 Msps × ~1018 samps/block → ~80 blocks ≈ 81k samples / 2 ms.
        // We cap to a sane upper bound to bound allocation in the pipeline.
        let target_samps = (sample_rate as usize / 500).clamp(16_384, 131_072);
        let blocks_per_call = (target_samps + block_samps - 1) / block_samps;
        let max_samps = blocks_per_call * block_samps;

        let r = unsafe { skiq_start_rx_streaming(card, hdl) };
        if r != 0 {
            teardown(card, false);
            return Err(format!("skiq_start_rx_streaming failed: {}", r));
        }

        log::info!(
            "sidekiq open (card={}, hdl={}, port={}, {} MHz, {} MS/s, \
             gain_idx={} of {}..{}, block={} samps, batch={} blocks = {} samps)",
            card,
            hdl_name(hdl),
            port_name,
            center_freq / 1_000_000,
            sample_rate / 1_000_000,
            gi,
            gmin,
            gmax,
            block_samps,
            blocks_per_call,
            max_samps,
        );

        let _ = block_samps; // retained as a local for the open-time log message
        Ok(SidekiqHandle {
            card,
            hdl,
            max_samps,
            enabled: true,
            streaming: true,
            overflow_count: 0,
            drop_count: 0,
            iq_shift,
            last_ts_next: None,
            running: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Receive complex samples into an interleaved i16 buffer.
    ///
    /// libsidekiq delivers one ~1k-sample block per `skiq_receive` call. We
    /// aggregate multiple blocks per call so the downstream PFB/burst detector
    /// sees long enough contiguous chunks (~2 ms at the configured rate) to
    /// stitch BLE packets without losing continuity across block boundaries.
    ///
    /// Each block's `rf_timestamp` is checked against the expected next-block
    /// timestamp; any gap signals dropped samples (USB or internal-queue
    /// starvation) and we return early so the demod sees the discontinuity
    /// at a packet boundary rather than mid-packet.
    ///
    /// Sample format: 12-bit signed in (Q, I) order in each block. We swap to
    /// interleaved (I, Q) and shift-left-by-4 to fill the i16 range, matching
    /// the bladeRF SC16_Q11 convention so blue-dragon's squelch/FFT-power
    /// calibration carries over directly.
    pub fn recv_into_i16(&mut self, buf: &mut [i16]) -> usize {
        if !self.streaming {
            return 0;
        }

        let max_pairs = (buf.len() / 2).min(self.max_samps);
        if max_pairs == 0 {
            return 0;
        }

        let mut filled: usize = 0;
        let mut no_data_spins: u32 = 0;

        while filled < max_pairs {
            let mut hdl_out: c_int = self.hdl;
            let mut pblk: *mut c_void = ptr::null_mut();
            let mut len: u32 = 0;

            let st = unsafe { skiq_receive(self.card, &mut hdl_out, &mut pblk, &mut len) };
            if st == SKIQ_RX_STATUS_NO_DATA {
                no_data_spins += 1;
                if no_data_spins > 1024 {
                    break;
                }
                continue;
            } else if st != SKIQ_RX_STATUS_SUCCESS
                || pblk.is_null()
                || len <= SKIQ_RX_HEADER_SIZE_IN_BYTES
            {
                log::warn!("sidekiq: skiq_receive status={} len={}", st, len);
                break;
            }
            no_data_spins = 0;

            let payload_bytes = len - SKIQ_RX_HEADER_SIZE_IN_BYTES;
            let blk_pairs = (payload_bytes / 4) as usize;
            let want = (max_pairs - filled).min(blk_pairs);

            unsafe {
                // Header layout (24 bytes):
                //   [0..8]   rf_timestamp  (u64 LE)
                //   [8..16]  sys_timestamp (u64 LE)
                //   [16..24] meta bitfield (rfic_control etc.)
                let blk_u8 = pblk as *const u8;
                let rf_ts = ptr::read_unaligned(blk_u8 as *const u64);
                let meta = ptr::read_unaligned(blk_u8.add(16) as *const u64);

                if (meta >> 6) & 0x1 != 0 {
                    self.overflow_count = self.overflow_count.wrapping_add(1);
                }

                // Detect sample drops between consecutive blocks. The next
                // block's rf_timestamp should equal the previous block's
                // rf_timestamp + previous block's sample count.
                if let Some(expected) = self.last_ts_next {
                    if rf_ts != expected {
                        let gap = rf_ts.wrapping_sub(expected) as i64;
                        log::warn!(
                            "sidekiq: rf_timestamp gap {} samples (expected {}, got {}) -- \
                             returning partial buffer to preserve packet boundary alignment",
                            gap, expected, rf_ts
                        );
                        self.drop_count = self.drop_count.wrapping_add(1);
                        self.last_ts_next = Some(rf_ts.wrapping_add(blk_pairs as u64));
                        // Don't consume this block's samples into a buffer that
                        // already has prior-timeline samples in it.
                        if filled > 0 {
                            return filled;
                        }
                    }
                }
                self.last_ts_next = Some(rf_ts.wrapping_add(blk_pairs as u64));

                let data_ptr = blk_u8.add(SKIQ_RX_HEADER_SIZE_IN_BYTES as usize) as *const i16;
                let out_base = filled * 2;
                let sh = self.iq_shift;
                for i in 0..want {
                    let q = ptr::read_unaligned(data_ptr.add(2 * i));
                    let iv = ptr::read_unaligned(data_ptr.add(2 * i + 1));
                    buf[out_base + 2 * i] = iv.wrapping_shl(sh);
                    buf[out_base + 2 * i + 1] = q.wrapping_shl(sh);
                }
            }
            filled += want;
        }

        filled
    }

    /// Set RX gain index at runtime (manual gain mode).
    pub fn set_gain(&self, gain: f64) {
        unsafe {
            let mut gmin: u8 = 0;
            let mut gmax: u8 = 76;
            let _ = skiq_read_rx_gain_index_range(self.card, self.hdl, &mut gmin, &mut gmax);
            let gi = (gain.round() as i32).clamp(gmin as i32, gmax as i32) as u8;
            let _ = skiq_write_rx_gain(self.card, self.hdl, gi);
        }
    }

    pub fn max_samps(&self) -> usize {
        self.max_samps
    }

    /// RFIC overload (input saturation) count — the "overload" bit in the
    /// receive-block header. Separate from sample drops; useful to tell apart
    /// "lower the gain" (overload firing) vs "we lost samples" (drops firing).
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Detected sample-drop count from rf_timestamp gaps between consecutive
    /// receive blocks. Non-zero indicates the chain (FPGA DMA queue, PCIe,
    /// recv loop, or downstream pipeline) is not keeping up, distinct from
    /// RFIC saturation.
    pub fn drop_count(&self) -> u64 {
        self.drop_count
    }
}

impl Drop for SidekiqHandle {
    fn drop(&mut self) {
        unsafe {
            if self.streaming {
                skiq_stop_rx_streaming(self.card, self.hdl);
            }
            if self.enabled {
                skiq_disable_cards(&self.card, 1);
            }
        }
        sdk_release();
    }
}

fn hdl_name(h: c_int) -> &'static str {
    match h {
        x if x == SKIQ_RX_HDL_A1 => "A1",
        x if x == SKIQ_RX_HDL_A2 => "A2",
        x if x == SKIQ_RX_HDL_B1 => "B1",
        x if x == SKIQ_RX_HDL_B2 => "B2",
        _ => "?",
    }
}

fn port_name(p: c_int) -> &'static str {
    match p {
        x if x == SKIQ_RF_PORT_J1 => "J1(RX1)",
        x if x == SKIQ_RF_PORT_J2 => "J2(TX/RX2)",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// SdrSource implementation (channel-based path; mirrors bladerf.rs).
// ---------------------------------------------------------------------------

pub struct SidekiqSource {
    iface: String,
    sample_rate: u32,
    center_freq: u64,
    gain: f64,
    antenna: Option<String>,
    running: Arc<AtomicBool>,
}

impl SidekiqSource {
    pub fn new(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: f64,
        antenna: Option<&str>,
    ) -> Result<Self, String> {
        Ok(Self {
            iface: iface.to_string(),
            sample_rate,
            center_freq,
            gain,
            antenna: antenna.map(str::to_string),
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }
}

impl SdrSource for SidekiqSource {
    fn start(&mut self, tx: Sender<SampleBuf>) -> Result<(), String> {
        let mut h = SidekiqHandle::open(
            &self.iface,
            self.sample_rate,
            self.center_freq,
            self.gain,
            self.antenna.as_deref(),
        )?;
        self.running.store(true, Ordering::SeqCst);

        // Reuse a single scratch buffer across iterations to avoid an
        // allocation on every block aggregate; samples are then moved into
        // an owned Vec for SampleBuf.
        let max_samps = h.max_samps();
        let mut scratch = vec![0i16; max_samps * 2];

        while self.running.load(Ordering::SeqCst) {
            let pairs = h.recv_into_i16(&mut scratch);
            if pairs == 0 {
                continue;
            }
            let len = pairs * 2;
            let mut data = Vec::with_capacity(len);
            data.extend_from_slice(&scratch[..len]);
            if tx
                .send(SampleBuf {
                    data,
                    num_samples: pairs,
                })
                .is_err()
            {
                break;
            }
        }

        drop(h);
        log::info!("sidekiq streaming stopped");
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

/// Best-effort: lower the recv thread's nice value so PCIe block reads keep
/// up under load. Quietly no-ops if we lack CAP_SYS_NICE (the default for
/// unprivileged builds), which is the safe failure mode -- the recv loop
/// still works, it's just more susceptible to scheduling jitter at high
/// sample rates.
fn boost_recv_thread() {
    extern "C" {
        fn setpriority(which: c_int, who: c_uint, prio: c_int) -> c_int;
    }
    const PRIO_PROCESS: c_int = 0;
    // -10 is conservative; we don't want to starve the rest of the pipeline.
    unsafe {
        let _ = setpriority(PRIO_PROCESS, 0, -10);
    }
}

// Quiet unused-import warnings when only one path of the module is exercised.
#[allow(dead_code)]
fn _ffi_keepalive() {
    let _ = SKIQ_XPORT_TYPE_PCIE;
}
