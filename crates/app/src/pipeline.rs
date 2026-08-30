// Copyright 2025-2026 CEMAXECUTER LLC

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
#[cfg(feature = "zmq")]
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crossbeam::channel;
use num_complex::Complex32;

use bd_dsp::burst::BurstCatcher;
use bd_dsp::fft::BatchFft;
use bd_dsp::fsk::{self, FskDemod, FskResult};
use bd_dsp::pfb::PfbChannelizer;
use bd_dsp::window;
use bd_output::pcap::PcapWriter;
use bd_protocol::ble::{self, AaCorrelator};
use bd_protocol::ble_connection::ConnectionTable;
use bd_protocol::btbb::{self, SyndromeMap};
use bd_protocol::Timespec;
use bd_sdr::file::{FileSource, SampleFormat};
use bd_sdr::SdrSource;

use crate::burst_file::{FileBurstReader, FileBurstWriter};

fn open_burst_writer(
    path: Option<&Path>,
    max_bytes: u64,
) -> Result<Option<FileBurstWriter>, String> {
    path.map(|path| {
        FileBurstWriter::create(path, max_bytes)
            .map_err(|e| format!("failed to create {}: {}", path.display(), e))
    })
    .transpose()
}

fn record_burst(writer: &mut Option<FileBurstWriter>, burst: &bd_dsp::burst::Burst) {
    let result = match writer.as_mut() {
        Some(writer) => writer.write_burst(burst),
        None => return,
    };
    match result {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("burst capture size limit reached; recording stopped");
            *writer = None;
        }
        Err(e) => {
            eprintln!("burst capture write error: {}; recording stopped", e);
            *writer = None;
        }
    }
}

fn seed_classic_uaps(tracker: &mut btbb::PiconetTracker, classic_uaps: &[(u32, u8)]) {
    for &(lap, uap) in classic_uaps {
        tracker.set_uap(lap, uap);
        eprintln!("Classic ground truth: UAP={:02X} LAP={:06X}", uap, lap);
    }
}

/// Run the full pipeline from IQ file to PCAP output.
pub fn run_file(
    file_path: &Path,
    format: SampleFormat,
    center_freq_mhz: u32,
    num_channels: usize,
    pcap_path: Option<&Path>,
    burst_path: Option<&Path>,
    burst_limit_bytes: u64,
    check_crc: bool,
    squelch_db: f32,
    print_stats: bool,
    classic_uaps: &[(u32, u8)],
) -> Result<(), String> {
    let sample_rate = num_channels as u32 * 1_000_000; // 1 MHz per channel
    let center_freq_hz = center_freq_mhz as u64 * 1_000_000;

    // Build channel frequency table (matching C FFT bin ordering)
    // Bin 0 = center freq, bins 1..M/2-1 = center+1..center+M/2-1
    // Bins M/2..M-1 = center-M/2..center-1
    let channel_freqs: Vec<u32> = (0..num_channels)
        .map(|i| {
            if i < num_channels / 2 {
                center_freq_mhz + i as u32
            } else {
                (center_freq_mhz as i32 - num_channels as i32 + i as i32) as u32
            }
        })
        .collect();

    // Map FFT bins to valid BLE channels (even MHz, 2402-2480).
    // live_ch[ble_channel] = fft_bin_index, where ble_channel = (freq - 2402) / 2
    let mut live_ch: [i32; 40] = [-1; 40];
    let mut first_live: usize = 40;
    let mut last_live: usize = 0;
    for (fft_bin, &freq) in channel_freqs.iter().enumerate() {
        if freq >= 2402 && freq <= 2480 && (freq & 1) == 0 {
            let ch_num = ((freq - 2402) / 2) as usize;
            live_ch[ch_num] = fft_bin as i32;
            if ch_num < first_live {
                first_live = ch_num;
            }
            if ch_num > last_live {
                last_live = ch_num;
            }
        }
    }

    if first_live > last_live {
        return Err("no valid BLE channels in frequency range".to_string());
    }

    let active_channels = (first_live..=last_live)
        .filter(|&ch| live_ch[ch] >= 0)
        .count();
    let active_bt_channels = channel_freqs
        .iter()
        .filter(|&&freq| (2402..=2480).contains(&freq))
        .count();

    eprintln!(
        "channels: {} FFT bins, {} Classic + {} BLE channels (ch {}-{}, {}-{} MHz)",
        num_channels,
        active_bt_channels,
        active_channels,
        first_live,
        last_live,
        2402 + first_live * 2,
        2402 + last_live * 2,
    );

    // Initialize protocol subsystems
    let aa_correlator = AaCorrelator::new(); // LE 1M: SPS=2
    let aa_correlator_2m = AaCorrelator::with_sps(1); // LE 2M: SPS=1
    let syndrome_map = SyndromeMap::new(1);
    let mut conn_table = ConnectionTable::new();
    let mut bt_tracker = btbb::PiconetTracker::new();
    seed_classic_uaps(&mut bt_tracker, classic_uaps);
    let mut smp_parser = bd_protocol::smp::SmpParser::new();

    // Initialize DSP -- Type 2 PFB matching C code (semi_len m=4)
    let semi_len = 4;
    let prototype = window::pfb_prototype_float(num_channels, semi_len);
    let mut channelizer = PfbChannelizer::new(num_channels, semi_len, &prototype);
    let mut fft = BatchFft::new(num_channels);
    let sps = 2usize; // 2 samples per symbol (type 2 PFB output rate)

    // One catcher per in-band Classic channel (every MHz, 2402-2480).
    let mut burst_catchers: Vec<Option<BurstCatcher>> = channel_freqs
        .iter()
        .map(|&freq| {
            if (2402..=2480).contains(&freq) {
                Some(BurstCatcher::new(freq, squelch_db))
            } else {
                None
            }
        })
        .collect();

    // FSK demodulator
    let mut fsk = FskDemod::new(sps);

    // PCAP writer
    let mut pcap_writer: Option<PcapWriter<BufWriter<File>>> = if let Some(path) = pcap_path {
        let file = File::create(path)
            .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
        let writer = BufWriter::new(file);
        Some(PcapWriter::new(writer).map_err(|e| format!("failed to write PCAP header: {}", e))?)
    } else {
        None
    };
    let mut burst_writer = open_burst_writer(burst_path, burst_limit_bytes)?;

    // Stats
    let mut total_ble: u64 = 0;
    let mut total_bt: u64 = 0;
    let mut total_crc: u64 = 0;
    let mut valid_crc: u64 = 0;
    let mut total_bursts: u64 = 0;
    let mut pcap_errors: u64 = 0;
    let stats_start = Instant::now();
    let mut last_stats = Instant::now();

    // File source
    let mut source = FileSource::new(
        file_path.to_string_lossy().to_string(),
        format,
        sample_rate,
        center_freq_hz,
    );

    let (tx, rx) = channel::bounded(64);

    let reader_thread = std::thread::spawn(move || {
        if let Err(e) = source.start(tx) {
            log::error!("file reader error: {}", e);
        }
    });

    // Timestamp counter
    let mut sample_count: u64 = 0;
    let samples_to_timespec = |count: u64, rate: u32| -> Timespec {
        let secs = count / rate as u64;
        let frac = count % rate as u64;
        let nsec = frac * 1_000_000_000 / rate as u64;
        Timespec {
            tv_sec: secs,
            tv_nsec: nsec,
        }
    };

    // FFT normalization factor (matching C's agc_submit: / channels)
    let fft_scale = 1.0 / num_channels as f32;

    // Pre-allocated buffers (avoid per-step allocation)
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); num_channels];

    // Raw wideband ring buffer for phase-preserving EDR demod. An EDR packet's
    // DPSK payload is ~1.4 MHz wide and the ~1 MHz PFB channel clips it, so we
    // keep recent raw samples and re-extract the channel at full bandwidth when
    // an EDR-typed packet is detected. Bounded to a few tens of milliseconds.
    let edr_wideband_enabled = std::env::var_os("BD_EDR_WIDEBAND").is_some();
    let raw_ring_cap: usize = (sample_rate as usize / 25).max(1 << 16); // ~40 ms
    let mut raw_ring: Vec<Complex32> = Vec::with_capacity(raw_ring_cap + (1 << 16));
    let mut raw_ring_start: u64 = 0; // raw complex index of raw_ring[0]

    // Main processing loop
    for buf in rx.iter() {
        // Type 2 PFB: each call takes M int16 values (M/2 complex samples)
        let step = num_channels; // M int16 values per PFB call
        let num_blocks = buf.data.len() / step;

        // Append this block's raw complex samples to the wideband ring buffer
        // (only when the opt-in wideband EDR path is active).
        if edr_wideband_enabled {
            raw_ring_start += trim_ring(&mut raw_ring, raw_ring_cap);
            for i in 0..buf.num_samples {
                raw_ring.push(Complex32::new(buf.data[2 * i] as f32, buf.data[2 * i + 1] as f32));
            }
        }

        for block in 0..num_blocks {
            let offset = block * step;
            let end = offset + step;
            if end > buf.data.len() {
                break;
            }

            // PFB channelize + FFT (pre-allocated buffer, no alloc per step)
            channelizer.execute_into(&buf.data[offset..end], &mut fft_buf);
            fft.process(&mut fft_buf);

            // Normalize by 1/M (matching C's agc_submit division)
            for val in fft_buf.iter_mut() {
                *val *= fft_scale;
            }

            let ts = samples_to_timespec(sample_count, sample_rate);

            // Feed every in-band Classic channel; even-MHz bins also carry BLE.
            for fft_bin in 0..num_channels {
                let catcher = match burst_catchers[fft_bin].as_mut() {
                    Some(c) => c,
                    None => continue,
                };

                let sample = fft_buf[fft_bin];
                if let Some(burst) = catcher.execute(sample, &ts) {
                    total_bursts += 1;
                    record_burst(&mut burst_writer, &burst);

                    // Skip very short bursts (< 132 samples, matching C)
                    if burst.samples.len() < 132 {
                        continue;
                    }

                    // Preserve the normal decoder's original input. Any EDR
                    // lead-in is reserved for the missed-header fallback.
                    let normal_samples =
                        &burst.samples[burst.edr_lead_samples.min(burst.samples.len())..];
                    if let Some(mut fsk_result) = fsk.demodulate(normal_samples) {
                        let freq = burst.freq;
                        let burst_ts = burst.timestamp.clone();
                        let rssi = burst.rssi_db as i32;
                        let noise = burst.noise_db as i32;
                        let raw_start = burst.timestamp.tv_sec as u64 * sample_rate as u64
                            + (burst.timestamp.tv_nsec as u64 * sample_rate as u64)
                                / 1_000_000_000;
                        let raw_len = (sample_rate as usize / 1000) * 3;
                        // Provide the raw wideband window for any burst (a cheap
                        // slice). The narrow-channel EDR envelope test misses
                        // packets whose 1 MHz bin clips the DPSK, so gating the
                        // window on burst.edr_extended alone drops real EDR; the
                        // EDR header candidates + payload CRC inside decide.
                        let wideband = if edr_wideband_enabled {
                            ring_window(&raw_ring, raw_ring_start, raw_start, raw_len)
                        } else {
                            None
                        };
                        let offset_hz =
                            (burst.freq as f64 - center_freq_mhz as f64) * 1_000_000.0;
                        let mut demod_offset = burst.edr_lead_samples;
                        let mut classic = btbb::detect(
                            &fsk_result.bits,
                            freq,
                            rssi,
                            noise,
                            burst_ts.clone(),
                            &syndrome_map,
                        );
                        if burst.edr_extended {
                            if let Some((fallback_fsk, fallback_packet, fallback_offset)) =
                                recover_edr_lead_header(
                                    &mut fsk,
                                    &burst,
                                    rssi,
                                    noise,
                                    &syndrome_map,
                                )
                            {
                                fsk_result = fallback_fsk;
                                classic = Some(fallback_packet);
                                demod_offset = fallback_offset;
                            }
                        }
                        // Re-decode the header from clean raw IQ when there is no
                        // header, or when the channelized header decoded as a
                        // multi-slot Basic Rate type: a squelch-clipped EDR header
                        // reads that way and yields no EDR candidate with the
                        // verified UAP.
                        let header_needs_raw = classic.as_ref().is_none_or(|packet| {
                            !packet.has_header
                                || packet.header.is_some_and(|h| {
                                    btbb::edr_bits_per_symbol(h.pkt_type).is_none()
                                        && matches!(h.pkt_type & 0x0f, 0x8 | 0x9 | 0xa | 0xb | 0xe | 0xf)
                                })
                        });
                        if header_needs_raw {
                            if let Some((fallback_fsk, fallback_packet, fallback_offset)) =
                                wideband.and_then(|window| {
                                    recover_edr_wideband_header(
                                        window,
                                        offset_hz,
                                        sample_rate,
                                        burst.freq,
                                        rssi,
                                        noise,
                                        &syndrome_map,
                                        burst.edr_lead_samples,
                                    )
                                })
                            {
                                fsk_result = fallback_fsk;
                                classic = Some(fallback_packet);
                                demod_offset = fallback_offset;
                            }
                        }

                        // Try Classic BT first
                        if let Some(mut bt_pkt) = classic {
                            let demod_ts = channel_samples_after(&burst_ts, demod_offset);
                            bt_pkt.timestamp =
                                bt_sync_timestamp(&demod_ts, &fsk_result, bt_pkt.sync_offset);
                            let mut announce = btbb::enrich(&mut bt_pkt, &mut bt_tracker);
                            if try_enrich_edr(
                                &burst,
                                &fsk_result,
                                &mut bt_pkt,
                                &mut bt_tracker,
                                wideband,
                                offset_hz,
                                sample_rate,
                                demod_offset,
                            )
                            .enriched
                            {
                                announce |= bt_tracker.mark_announced(bt_pkt.lap);
                            }
                            if announce {
                                log_bt_address(&bt_pkt);
                            }
                            total_bt += 1;
                            if bt_pkt.crc_ok {
                                total_crc += 1;
                                valid_crc += 1;
                            }
                            if let Some(ref mut writer) = pcap_writer {
                                if let Err(e) = writer.write_bt(&bt_pkt, None) {
                                    if pcap_errors == 0 {
                                        eprintln!("PCAP write error: {}", e);
                                    }
                                    pcap_errors += 1;
                                }
                            }
                        } else {
                            if freq & 1 != 0 {
                                continue;
                            }
                            let burst_len = fsk_result.demod.len();
                            let mut pkt = None;

                            // Try coded first on long bursts
                            if burst_len > 2000 {
                                pkt = ble::ble_coded_burst(
                                    &fsk_result.demod,
                                    freq,
                                    burst_ts.clone(),
                                    2, // SPS=2
                                    check_crc,
                                    |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
                                );
                            }

                            // Try BLE LE 1M preamble-first detection
                            if pkt.is_none() {
                                pkt = ble::ble_burst(
                                    &fsk_result.bits,
                                    freq,
                                    burst_ts.clone(),
                                    check_crc,
                                    |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
                                );
                            }

                            // Fall back to LE 1M AA correlator
                            if pkt.is_none() {
                                pkt = aa_correlator.correlate(
                                    &fsk_result.demod,
                                    freq,
                                    burst_ts.clone(),
                                    check_crc,
                                );
                            }

                            // Try LE 2M: reslice at SPS=1
                            if pkt.is_none() {
                                let bits_2m =
                                    fsk::reslice(&fsk_result.demod, fsk_result.silence, 1);
                                pkt = ble::ble_burst_2m(
                                    &bits_2m,
                                    freq,
                                    burst_ts.clone(),
                                    check_crc,
                                    |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
                                );

                                // Fall back to LE 2M AA correlator
                                if pkt.is_none() {
                                    pkt = aa_correlator_2m.correlate_2m(
                                        &fsk_result.demod,
                                        freq,
                                        burst_ts.clone(),
                                        check_crc,
                                    );
                                }
                            }

                            // Try coded on shorter bursts too (S=2 min ~2000 samples)
                            if pkt.is_none() && burst_len <= 2000 {
                                pkt = ble::ble_coded_burst(
                                    &fsk_result.demod,
                                    freq,
                                    burst_ts.clone(),
                                    2, // SPS=2
                                    check_crc,
                                    |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
                                );
                            }

                            if let Some(mut p) = pkt {
                                p.rssi_db = rssi;
                                p.noise_db = noise;

                                // Track CONNECT_IND
                                if p.aa == ble::BLE_ADV_AA && p.crc_valid {
                                    conn_table.parse_connect_ind(&p, burst_ts.tv_sec);
                                }

                                // Parse SMP from data channel packets
                                if p.aa != ble::BLE_ADV_AA && p.crc_valid && p.is_data {
                                    if let Some((cid, payload)) =
                                        bd_protocol::smp::extract_l2cap(&p.data)
                                    {
                                        for event in
                                            smp_parser.parse_l2cap(p.aa, cid, payload, true)
                                        {
                                            match &event {
                                                bd_protocol::smp::SmpEvent::WeakPairing {
                                                    aa,
                                                    reason,
                                                } => {
                                                    eprintln!(
                                                        "SMP WARNING: 0x{:08X}: {}",
                                                        aa, reason
                                                    );
                                                }
                                                bd_protocol::smp::SmpEvent::LtkDistributed {
                                                    aa,
                                                    ..
                                                } => {
                                                    eprintln!("SMP: 0x{:08X} LTK captured", aa);
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }

                                // Track CRC stats
                                if p.crc_checked {
                                    total_crc += 1;
                                    if p.crc_valid {
                                        valid_crc += 1;
                                    }
                                }

                                total_ble += 1;
                                if let Some(ref mut writer) = pcap_writer {
                                    if let Err(e) = writer.write_ble(&p, None) {
                                        if pcap_errors == 0 {
                                            eprintln!("PCAP write error: {}", e);
                                        }
                                        pcap_errors += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // M/2 complex samples consumed per PFB call
            sample_count += (num_channels / 2) as u64;
        }

        // Print stats every 5 seconds
        if print_stats && last_stats.elapsed().as_secs() >= 5 {
            let elapsed = stats_start.elapsed().as_secs_f64();
            let crc_pct = if total_crc > 0 {
                (valid_crc as f64 / total_crc as f64) * 100.0
            } else {
                0.0
            };
            let conns = conn_table.count();
            eprintln!(
                "[{:.1}s] BLE: {} BT: {} bursts: {} CRC: {:.1}% ({}/{}) conns: {}",
                elapsed, total_ble, total_bt, total_bursts, crc_pct, valid_crc, total_crc, conns,
            );
            last_stats = Instant::now();
        }
    }

    // Final stats
    if print_stats {
        let elapsed = stats_start.elapsed().as_secs_f64();
        let crc_pct = if total_crc > 0 {
            (valid_crc as f64 / total_crc as f64) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "done ({:.1}s): BLE: {} BT: {} bursts: {} CRC: {:.1}% ({}/{})",
            elapsed, total_ble, total_bt, total_bursts, crc_pct, valid_crc, total_crc,
        );
    }

    let _ = reader_thread.join();

    Ok(())
}

/// Replay compact channelized bursts without rerunning the wideband channelizer.
pub fn run_burst_file(
    path: &Path,
    pcap_path: Option<&Path>,
    check_crc: bool,
    print_stats: bool,
    classic_uaps: &[(u32, u8)],
) -> Result<(), String> {
    let mut reader = FileBurstReader::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let mut pcap_writer = if let Some(path) = pcap_path {
        let file = File::create(path)
            .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
        Some(
            PcapWriter::new(BufWriter::new(file))
                .map_err(|e| format!("failed to write PCAP header: {}", e))?,
        )
    } else {
        None
    };

    let mut fsk = FskDemod::new(2);
    let aa_correlator = AaCorrelator::new();
    let aa_correlator_2m = AaCorrelator::with_sps(1);
    let syndrome_map = SyndromeMap::new(1);
    let mut bt_tracker = btbb::PiconetTracker::new();
    seed_classic_uaps(&mut bt_tracker, classic_uaps);
    let mut conn_table = ConnectionTable::new();
    let mut smp_parser = bd_protocol::smp::SmpParser::new();
    let mut stats = PipelineStats::new();
    #[cfg(feature = "zmq")]
    let zmq_pub: Option<bd_output::zmq_pub::ZmqPublisher> = None;

    while let Some(burst) = reader
        .read_burst()
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?
    {
        process_burst(
            &burst,
            None,
            0,
            1,
            &mut fsk,
            &aa_correlator,
            &aa_correlator_2m,
            &syndrome_map,
            &mut bt_tracker,
            &mut conn_table,
            &mut smp_parser,
            &mut pcap_writer,
            #[cfg(feature = "zmq")]
            &zmq_pub,
            None,
            check_crc,
            &mut stats,
        );
    }

    if print_stats {
        eprintln!(
            "replay done: BLE: {} BT: {} bursts: {} CRC: {:.1}% ({}/{}) EDR: try={} sync={} crc={} best={:.3}",
            stats.total_ble,
            stats.total_bt,
            stats.total_bursts,
            stats.crc_pct(),
            stats.valid_crc,
            stats.total_crc,
            stats.edr_attempts,
            stats.edr_syncs,
            stats.edr_crc_matches,
            stats.edr_best_sync_score.unwrap_or(f32::NAN),
        );
    }
    Ok(())
}

/// Build channel frequency table and live channel mapping.
/// Returns (channel_freqs, live_ch, first_live, last_live).
fn build_channel_map(
    center_freq_mhz: u32,
    num_channels: usize,
) -> Result<(Vec<u32>, [i32; 40], usize, usize), String> {
    let channel_freqs: Vec<u32> = (0..num_channels)
        .map(|i| {
            if i < num_channels / 2 {
                center_freq_mhz + i as u32
            } else {
                (center_freq_mhz as i32 - num_channels as i32 + i as i32) as u32
            }
        })
        .collect();

    let mut live_ch: [i32; 40] = [-1; 40];
    let mut first_live: usize = 40;
    let mut last_live: usize = 0;
    for (fft_bin, &freq) in channel_freqs.iter().enumerate() {
        if freq >= 2402 && freq <= 2480 && (freq & 1) == 0 {
            let ch_num = ((freq - 2402) / 2) as usize;
            live_ch[ch_num] = fft_bin as i32;
            if ch_num < first_live {
                first_live = ch_num;
            }
            if ch_num > last_live {
                last_live = ch_num;
            }
        }
    }

    if first_live > last_live {
        return Err("no valid BLE channels in frequency range".to_string());
    }

    Ok((channel_freqs, live_ch, first_live, last_live))
}

/// Print a one-line note when a Classic BT device's address is first recovered.
fn log_bt_address(pkt: &bd_protocol::btbb::ClassicBtPacket) {
    let lap = pkt.lap;
    match (pkt.uap, pkt.nap) {
        (Some(uap), Some(nap)) => eprintln!(
            "BT addr {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X} (full, from FHS)",
            (nap >> 8) as u8,
            nap as u8,
            uap,
            (lap >> 16) as u8,
            (lap >> 8) as u8,
            lap as u8,
        ),
        (Some(uap), None) => eprintln!(
            "BT addr ??:??:{:02X}:{:02X}:{:02X}:{:02X}  UAP CRC-verified (LAP+UAP){}",
            uap,
            (lap >> 16) as u8,
            (lap >> 8) as u8,
            lap as u8,
            pkt.clkn
                .map(|c| format!("  CLK={:07X}", c))
                .unwrap_or_default(),
        ),
        _ => {}
    }
}

fn bt_sync_timestamp(
    burst_timestamp: &Timespec,
    fsk_result: &FskResult,
    sync_offset: usize,
) -> Timespec {
    const CHANNEL_SAMPLE_RATE: u64 = 2_000_000;
    const SPS: usize = 2;

    let sample_offset = fsk_result.silence + 1 + sync_offset * SPS;
    let offset_ns = sample_offset as u64 * 1_000_000_000 / CHANNEL_SAMPLE_RATE;
    let total_ns = burst_timestamp.tv_nsec + offset_ns;
    Timespec {
        tv_sec: burst_timestamp.tv_sec + total_ns / 1_000_000_000,
        tv_nsec: total_ns % 1_000_000_000,
    }
}

fn recover_edr_lead_header(
    fsk: &mut FskDemod,
    burst: &bd_dsp::burst::Burst,
    rssi: i32,
    noise: i32,
    syndrome_map: &SyndromeMap,
) -> Option<(FskResult, btbb::ClassicBtPacket, usize)> {
    if !burst.edr_extended || burst.edr_lead_samples < 16 {
        return None;
    }
    for offset in (0..burst.edr_lead_samples).step_by(8) {
        let Some(result) = fsk.demodulate(&burst.samples[offset..]) else {
            continue;
        };
        let Some(packet) = btbb::detect(
            &result.bits,
            burst.freq,
            rssi,
            noise,
            burst.timestamp.clone(),
            syndrome_map,
        ) else {
            continue;
        };
        if packet.has_header {
            if std::env::var_os("BD_EDR_DEBUG").is_some() {
                eprintln!(
                    "[edr-lead] freq={} offset={} LAP={:06X} access_errors={}",
                    burst.freq, offset, packet.lap, packet.ac_errors
                );
            }
            return Some((result, packet, offset));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn recover_edr_wideband_header(
    wideband: &[Complex32],
    offset_hz: f64,
    raw_rate: u32,
    freq: u32,
    rssi: i32,
    noise: i32,
    syndrome_map: &SyndromeMap,
    lead_samples: usize,
) -> Option<(FskResult, btbb::ClassicBtPacket, usize)> {
    let baseband_4m = bd_dsp::edr::extract_br_channel_4m(wideband, offset_hz, raw_rate);
    if baseband_4m.len() < 512 {
        return None;
    }
    if std::env::var_os("BD_EDR_DEBUG").is_some() {
        let search_end_4m = lead_samples
            .saturating_mul(2)
            .min(baseband_4m.len().saturating_sub(264));
        let mut probe = FskDemod::new(4);
        for offset in (0..=search_end_4m).step_by(8) {
            let Some(result) = probe.demodulate(&baseband_4m[offset..]) else {
                continue;
            };
            if let Some(packet) = btbb::detect(
                &result.bits,
                freq,
                rssi,
                noise,
                Timespec::default(),
                syndrome_map,
            ) {
                if packet.has_header {
                    eprintln!(
                        "[edr-raw4-header] freq={} offset={} LAP={:06X} access_errors={}",
                        freq, offset, packet.lap, packet.ac_errors
                    );
                    break;
                }
            }
        }
    }
    let baseband_2m = baseband_4m.iter().step_by(2).copied().collect::<Vec<_>>();
    let search_end = lead_samples.min(baseband_2m.len().saturating_sub(136));
    let mut fsk = FskDemod::new(2);
    for offset in (0..=search_end).step_by(4) {
        let Some(result) = fsk.demodulate(&baseband_2m[offset..]) else {
            continue;
        };
        let Some(packet) = btbb::detect(
            &result.bits,
            freq,
            rssi,
            noise,
            Timespec::default(),
            syndrome_map,
        ) else {
            continue;
        };
        if packet.has_header {
            if std::env::var_os("BD_EDR_DEBUG").is_some() {
                eprintln!(
                    "[edr-raw-header] freq={} offset={} LAP={:06X} access_errors={}",
                    freq, offset, packet.lap, packet.ac_errors
                );
            }
            return Some((result, packet, offset));
        }
    }
    None
}

#[derive(Default)]
struct EdrAttempt {
    candidate: bool,
    synchronized: bool,
    crc_match: bool,
    enriched: bool,
    best_sync_score: Option<f32>,
}

/// Drop the oldest samples once the ring exceeds its cap; returns how many were
/// removed so the caller can advance the ring's base sample index.
fn trim_ring(ring: &mut Vec<Complex32>, cap: usize) -> u64 {
    if ring.len() > cap {
        let drop = ring.len() - cap;
        ring.drain(..drop);
        drop as u64
    } else {
        0
    }
}

/// Extract the raw wideband window covering a burst from the ring buffer, given
/// the burst's start sample index and raw length. Returns `None` if the window
/// is not fully resident (e.g. trimmed away).
fn ring_window(
    ring: &[Complex32],
    ring_start: u64,
    raw_start: u64,
    raw_len: usize,
) -> Option<&[Complex32]> {
    let off = raw_start.checked_sub(ring_start)? as usize;
    let end = off.checked_add(raw_len)?;
    if end <= ring.len() {
        Some(&ring[off..end])
    } else if off < ring.len() {
        Some(&ring[off..])
    } else {
        None
    }
}

/// Decode EDR payloads from a raw wideband window around the packet. The fixed
/// sync chooses the payload boundary before CRC validation; CRC is never used
/// to search for symbol alignment.
fn edr_wideband_variants(
    window: &[Complex32],
    offset_hz: f64,
    raw_rate: u32,
    channel_sync_reference: usize,
) -> Vec<(usize, Vec<u8>, f32)> {
    const SPS: usize = 4;
    const MAX_SYNC_SCORE: f32 = 0.08;
    let mut baseband = bd_dsp::edr::extract_edr_channel_4m(window, offset_hz, raw_rate);
    if baseband.len() < 64 * SPS {
        return Vec::new();
    }
    let _ = channel_sync_reference;
    let two_pi = 2.0 * std::f32::consts::PI;
    let search = two_pi * (300_000.0 / 4_000_000.0);
    let step = two_pi * (5_000.0 / 4_000_000.0);
    // Estimate the CFO from the packet itself, not a header-derived reference
    // (which is unreliable when the squelch clipped the access code).
    let correction = bd_dsp::edr::refine_cfo(&baseband, SPS, search, step);
    bd_dsp::edr::derotate(&mut baseband, correction);
    let matched = bd_dsp::edr::rrc_matched_filter(&baseband, SPS, 0.4, 6);
    // The fixed sync is a strong 10-symbol pattern (real locks score ~0.005),
    // so scan the whole window with the strict threshold rather than trusting an
    // approximate reference.
    let Some(lock) = bd_dsp::edr::locate_edr_sync(
        &matched,
        matched.len() / 2,
        matched.len() / 2,
        SPS,
        MAX_SYNC_SCORE,
    ) else {
        return Vec::new();
    };

    // The payload starts a fixed distance after the sync, but the exact symbol
    // count (guard/reference) leaves it a symbol or two ambiguous; a mis-set
    // start shifts the whitening phase and hides the CRC. Emit the demod for a
    // few start offsets around the lock and let the CRC choose.
    let mut out = Vec::new();
    for &bps in &[2usize, 3usize] {
        for delta in -3i32..=3 {
            let mut shifted = lock;
            let ref_sample = lock.reference_sample as i32 + delta * SPS as i32;
            if ref_sample < 0 {
                continue;
            }
            shifted.reference_sample = ref_sample as usize;
            let bits = bd_dsp::edr::demod_dpsk_detrended_from_sync(&matched, shifted, SPS, bps);
            if !bits.is_empty() && !out.iter().any(|(b, existing, _)| *b == bps && existing == &bits) {
                out.push((bps, bits, lock.score));
            }
        }
    }
    if std::env::var_os("BD_EDR_DEBUG").is_some() {
        eprintln!(
            "[edr-wb] sync={} score={:.4} conjugated={} cfo={:.0}Hz bits={:?}",
            lock.reference_sample,
            lock.score,
            lock.conjugated,
            correction * 4_000_000.0 / two_pi,
            out.iter().map(|(bps, bits, _)| (*bps, bits.len())).collect::<Vec<_>>()
        );
    }
    out
}

fn try_enrich_edr(
    burst: &bd_dsp::burst::Burst,
    fsk_result: &FskResult,
    pkt: &mut bd_protocol::btbb::ClassicBtPacket,
    tracker: &mut btbb::PiconetTracker,
    wideband: Option<&[Complex32]>,
    offset_hz: f64,
    raw_rate: u32,
    channel_lead_samples: usize,
) -> EdrAttempt {
    if !pkt.has_header {
        return EdrAttempt::default();
    }
    // A 3-DHx EDR packet whose GFSK header decoded under the wrong clock reads
    // as a multi-slot Basic Rate type (DH3/DH5/DM3/DM5/AUX1). Only bail out for
    // short types that can never carry an EDR payload; the EDR header candidates
    // and the payload CRC below decide the rest. Without a wideband window there
    // is nothing new to try, so keep the old fast path.
    if wideband.is_none()
        && pkt.uap_verified
        && pkt
            .header
            .is_some_and(|header| btbb::edr_bits_per_symbol(header.pkt_type).is_none())
    {
        return EdrAttempt::default();
    }
    if let Some(header) = pkt.header {
        if btbb::edr_bits_per_symbol(header.pkt_type).is_none()
            && !matches!(header.pkt_type & 0x0f, 0x8 | 0x9 | 0xa | 0xb | 0xe | 0xf)
        {
            // Short BR types (ID/NULL/POLL/FHS/DM1/DH1/HV*) are never EDR.
            return EdrAttempt::default();
        }
    }
    let mut candidates = btbb::edr_header_candidates(&pkt.raw_header);
    if pkt.uap_verified {
        if let Some(uap) = pkt.uap {
            candidates.retain(|(candidate_uap, _)| *candidate_uap == uap);
        }
    }
    if let (Some(uap), Some(header)) = (pkt.uap, pkt.header) {
        if btbb::edr_bits_per_symbol(header.pkt_type).is_some() {
            candidates.insert(0, (uap, header));
        }
    }
    // A squelch-clipped EDR header can be corrupt enough that no clock yields an
    // EDR candidate with the verified access-code UAP. The UAP itself is
    // reliable (it comes from the access code, not the header), so when we have a
    // wideband window fall back to every clock under that UAP and let the payload
    // CRC pick the real one. Wideband-only, so Basic Rate is unaffected.
    if wideband.is_some() {
        if let Some(uap) = pkt.uap.filter(|_| pkt.uap_verified) {
            if !candidates.iter().any(|(candidate_uap, _)| *candidate_uap == uap) {
                // One candidate is enough: the wideband path searches all
                // whitening phases, so the clock field is not used for it.
                candidates.push((
                    uap,
                    btbb::BtHeader {
                        lt_addr: 1,
                        pkt_type: 0x0d,
                        flow: 1,
                        arqn: 0,
                        seqn: 0,
                        hec: 0,
                        clk6: 0,
                    },
                ));
            }
        }
    }
    if candidates.is_empty() {
        return EdrAttempt::default();
    }
    if std::env::var_os("BD_EDR_DEBUG").is_some() {
        let corrected = btbb::edr_header_candidates_corrected(&pkt.raw_header);
        eprintln!(
            "[edr-wb] LAP={:06X} burst={}.{:09} raw_header={:02X?} headers=[{}] corrected=[{}]",
            pkt.lap,
            burst.timestamp.tv_sec,
            burst.timestamp.tv_nsec,
            pkt.raw_header,
            candidates
                .iter()
                .map(|(uap, header)| format!(
                    "{:02X}/clk{}/{}",
                    uap,
                    header.clk6,
                    btbb::pkt_type_name(header.pkt_type)
                ))
                .collect::<Vec<_>>()
                .join(","),
            corrected
                .iter()
                .filter(|(uap, _)| !pkt.uap_verified || pkt.uap == Some(*uap))
                .map(|(uap, header)| format!(
                    "{:02X}/clk{}/{}",
                    uap,
                    header.clk6,
                    btbb::pkt_type_name(header.pkt_type)
                ))
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    const SPS: usize = 2;
    const ACCESS_TRAILER_BITS: usize = 68;
    const CODED_HEADER_BITS: usize = 54;
    const EDR_GUARD_SYMBOLS: usize = 5;
    let sync_reference_symbol =
        pkt.sync_offset + ACCESS_TRAILER_BITS + CODED_HEADER_BITS + EDR_GUARD_SYMBOLS;
    let sync_reference_sample =
        channel_lead_samples + fsk_result.silence + 1 + sync_reference_symbol * SPS;
    let wb_variants = wideband
        .map(|window| {
            edr_wideband_variants(window, offset_hz, raw_rate, sync_reference_sample)
        })
        .unwrap_or_default();
    let mut attempt = EdrAttempt {
        candidate: true,
        ..EdrAttempt::default()
    };
    let iq = bd_dsp::edr::prepare_iq(
        &burst.samples,
        fsk_result.cfo * std::f32::consts::PI,
        fsk_result.resample_ratio,
    );
    let matched_iq = bd_dsp::edr::rrc_matched_filter(&iq, SPS, 0.4, 6);
    let mut corrected_candidates = None;
    let mut matches: Vec<(bd_protocol::btbb::ClassicBtPacket, u8, btbb::BtHeader)> = Vec::new();
    for bits_per_symbol in [2usize, 3] {
        if !candidates
            .iter()
            .any(|(_, header)| btbb::edr_bits_per_symbol(header.pkt_type) == Some(bits_per_symbol))
        {
            continue;
        }
        let crc_candidates = corrected_candidates.get_or_insert_with(|| {
            let mut corrected = btbb::edr_header_candidates_corrected(&pkt.raw_header);
            if pkt.uap_verified {
                if let Some(uap) = pkt.uap {
                    corrected.retain(|(candidate_uap, _)| *candidate_uap == uap);
                }
            }
            corrected
        });
        for (filter_name, demod_iq) in [("rrc", &matched_iq), ("raw", &iq)] {
            let (payload_variants, diagnostic) =
                bd_dsp::edr::demod_payload_variants_with_diagnostic(
                    demod_iq,
                    sync_reference_sample,
                    SPS,
                    bits_per_symbol,
                );
            if let Some(diagnostic) = diagnostic {
                if attempt
                    .best_sync_score
                    .is_none_or(|score| diagnostic.score < score)
                {
                    attempt.best_sync_score = Some(diagnostic.score);
                }
            }
            attempt.synchronized |= !payload_variants.is_empty();
            if !payload_variants.is_empty() {
                if let Some(diagnostic) = diagnostic {
                    let header_summary = crc_candidates
                        .iter()
                        .map(|(uap, header)| {
                            format!(
                                "{:02X}/clk{:02}/{}",
                                uap,
                                header.clk6,
                                btbb::pkt_type_name(header.pkt_type)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    let payload_summary = payload_variants
                        .iter()
                        .enumerate()
                        .flat_map(|(variant, bits)| {
                            crc_candidates.iter().filter_map(move |(_, header)| {
                                if btbb::edr_bits_per_symbol(header.pkt_type)
                                    != Some(bits_per_symbol)
                                {
                                    return None;
                                }
                                btbb::edr_payload_header(bits, header.clk6).map(|payload| {
                                    format!(
                                        "v{}({})/clk{}:{}/{}/{}",
                                        variant,
                                        bits.len(),
                                        header.clk6,
                                        payload.llid,
                                        payload.flow,
                                        payload.length
                                    )
                                })
                            })
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    log::debug!(
                        "EDR sync LAP={:06X} freq={} bps={} filter={} score={:.4} offset={} conjugated={} variants={} headers=[{}] payloads=[{}]",
                        pkt.lap,
                        pkt.freq,
                        bits_per_symbol,
                        filter_name,
                        diagnostic.score,
                        diagnostic.offset,
                        diagnostic.conjugated,
                        payload_variants.len(),
                        header_summary,
                        payload_summary,
                    );
                }
            }
            for raw_payload in payload_variants {
                for &(uap, header) in crc_candidates.iter() {
                    if btbb::edr_bits_per_symbol(header.pkt_type) != Some(bits_per_symbol) {
                        continue;
                    }
                    let mut candidate = pkt.clone();
                    if btbb::enrich_edr_candidate(&mut candidate, &raw_payload, uap, header)
                        && !matches.iter().any(|(_, seen_uap, seen_header)| {
                            *seen_uap == uap
                                && seen_header.clk6 == header.clk6
                                && seen_header.pkt_type == header.pkt_type
                        })
                    {
                        matches.push((candidate, uap, header));
                    }
                }
            }
        }

        // Merge phase-preserving wideband variants for this modulation. The
        // channelized path above works from a ~1 MHz PFB bin that clips the DPSK;
        // the wideband path recovers the full constellation. The CRC below is
        // still the sole authority for accepting a payload.
        for (wb_bps, raw_payload, sync_score) in &wb_variants {
            if *wb_bps != bits_per_symbol {
                continue;
            }
            attempt.synchronized = true;
            if attempt
                .best_sync_score
                .is_none_or(|best| *sync_score < best)
            {
                attempt.best_sync_score = Some(*sync_score);
            }
            // Wideband acceptance uses only uncorrected, HEC-consistent header
            // interpretations. One-bit FEC repair multiplied by timing variants
            // produced enough trials for accidental 16-bit CRC matches.
            //
            // The header's 4-bit TYPE field is unreliable here: the squelch often
            // rises on the DPSK payload after the GFSK header has passed, so the
            // recovered type can disagree with the modulation the signal actually
            // carries (e.g. a clock's header reads 2-DH3 while the payload is
            // 8DPSK). The whitening only needs the clock, and the modulation is
            // fixed by which bits_per_symbol variant this is, so pair each unique
            // HEC-consistent clock with a synthetic header of the matching rate
            // and let the payload CRC decide.
            // Whitening depends only on the clock; the type field is unreliable
            // after a clipped header, so pair each unique UAP with a synthetic
            // header of the right rate and search all whitening phases. The
            // payload CRC is the sole authority.
            let synth_type: u8 = if bits_per_symbol == 3 { 0x0d } else { 0x0c };
            let mut tried_uaps: Vec<u8> = Vec::new();
            for &(uap, base_header) in &candidates {
                if tried_uaps.contains(&uap) {
                    continue;
                }
                tried_uaps.push(uap);
                let header = btbb::BtHeader {
                    pkt_type: synth_type,
                    ..base_header
                };
                let mut candidate = pkt.clone();
                if btbb::enrich_edr_candidate_any_phase(&mut candidate, raw_payload, uap, header)
                    && (2..=1024).contains(&candidate.decoded_payload.len())
                    && !matches
                        .iter()
                        .any(|(_, seen_uap, _)| *seen_uap == uap)
                {
                    if std::env::var_os("BD_EDR_DEBUG").is_some() {
                        let head: String = candidate
                            .decoded_payload
                            .iter()
                            .take(24)
                            .map(|b| format!("{:02x} ", b))
                            .collect();
                        eprintln!(
                            "[edr-wb] CRC PASS LAP={:06X} UAP={:02X} bps={} bytes={} | {}",
                            pkt.lap, uap, bits_per_symbol, candidate.decoded_payload.len(), head,
                        );
                    }
                    matches.push((candidate, uap, header));
                }
            }
        }
    }

    let [(candidate, uap, header)] = matches.as_slice() else {
        return attempt;
    };
    attempt.crc_match = true;
    if tracker.confirm_uap_at(pkt.lap, *uap, header.clk6, &pkt.timestamp) {
        *pkt = candidate.clone();
        attempt.enriched = true;
    }
    attempt
}

/// Process a burst: FSK demod -> BLE/BT detect -> PCAP write + ZMQ publish.
#[allow(clippy::too_many_arguments)]
fn process_burst(
    burst: &bd_dsp::burst::Burst,
    wideband: Option<&[Complex32]>,
    center_freq_mhz: u32,
    raw_sample_rate: u32,
    fsk: &mut FskDemod,
    aa_correlator: &AaCorrelator,
    aa_correlator_2m: &AaCorrelator,
    syndrome_map: &SyndromeMap,
    bt_tracker: &mut btbb::PiconetTracker,
    conn_table: &mut ConnectionTable,
    smp_parser: &mut bd_protocol::smp::SmpParser,
    pcap_writer: &mut Option<PcapWriter<BufWriter<File>>>,
    #[cfg(feature = "zmq")] zmq_pub: &Option<bd_output::zmq_pub::ZmqPublisher>,
    gps_fix: Option<&bd_output::pcap::GpsFix>,
    check_crc: bool,
    stats: &mut PipelineStats,
) {
    // Scan bursts: coded PHY search only (no BT/BLE 1M/2M)
    if burst.scan {
        let raw_demod = fsk.fm_discriminate_raw(&burst.samples);
        stats.coded_attempts += 1;
        if let Some(mut p) = ble::ble_coded_burst_search(
            &raw_demod,
            burst.freq,
            burst.timestamp.clone(),
            2,
            check_crc,
            &mut |aa| conn_table.crc_init_for_aa(aa, burst.timestamp.tv_sec),
            raw_demod.len(), // search full scan window
        ) {
            p.rssi_db = burst.rssi_db as i32;
            p.noise_db = burst.noise_db as i32;
            // Require valid CRC for coded packets from scan to avoid
            // false positives (garbled AA/MAC = phantom devices)
            if p.crc_checked && !p.crc_valid {
                return;
            }
            if p.crc_checked {
                stats.total_crc += 1;
                if p.crc_valid {
                    stats.valid_crc += 1;
                }
            }
            stats.total_ble += 1;
            stats.total_ble_coded += 1;
            if let Some(ref mut writer) = pcap_writer {
                if let Err(e) = writer.write_ble(&p, gps_fix) {
                    if stats.pcap_errors == 0 {
                        eprintln!("PCAP write error: {}", e);
                    }
                    stats.pcap_errors += 1;
                }
            }
            #[cfg(feature = "zmq")]
            if let Some(ref pub_socket) = zmq_pub {
                pub_socket.send_ble(&p, gps_fix);
            }
        }
        return;
    }

    stats.total_bursts += 1;
    let blen = burst.samples.len();
    if blen > stats.max_burst_len {
        stats.max_burst_len = blen;
    }
    match blen {
        0..=199 => stats.burst_lt200 += 1,
        200..=999 => stats.burst_200_1k += 1,
        1000..=4999 => stats.burst_1k_5k += 1,
        5000..=49999 => stats.burst_5k_50k += 1,
        _ => stats.burst_50k_plus += 1,
    }
    if blen < 132 {
        return;
    }

    let normal_samples = &burst.samples[burst.edr_lead_samples.min(burst.samples.len())..];
    let fsk_result = match fsk.demodulate(normal_samples) {
        Some(r) => r,
        None => {
            // FSK demod failed. Try raw FM + coded for any burst >= 1500 samples
            // (coded S=2 min ~2000 samples, S=8 min ~4000 samples).
            if blen >= 1500 {
                stats.fsk_reject_long += 1;
                let raw_demod = fsk.fm_discriminate_raw(&burst.samples);
                stats.coded_attempts += 1;
                let search_depth = 0; // 0 = search full burst (coarse stride handles perf)
                if let Some(mut p) = ble::ble_coded_burst_search(
                    &raw_demod,
                    burst.freq,
                    burst.timestamp.clone(),
                    2,
                    check_crc,
                    &mut |aa| conn_table.crc_init_for_aa(aa, burst.timestamp.tv_sec),
                    search_depth,
                ) {
                    p.rssi_db = burst.rssi_db as i32;
                    p.noise_db = burst.noise_db as i32;
                    // Require valid CRC for coded packets from noise bursts
                    // to avoid false positives (garbled AA/MAC = phantom devices)
                    if p.crc_checked && !p.crc_valid {
                        return;
                    }
                    if p.crc_checked {
                        stats.total_crc += 1;
                        if p.crc_valid {
                            stats.valid_crc += 1;
                        }
                    }
                    stats.total_ble += 1;
                    stats.total_ble_coded += 1;
                    if let Some(ref mut writer) = pcap_writer {
                        if let Err(e) = writer.write_ble(&p, gps_fix) {
                            if stats.pcap_errors == 0 {
                                eprintln!("PCAP write error: {}", e);
                            }
                            stats.pcap_errors += 1;
                        }
                    }
                    #[cfg(feature = "zmq")]
                    if let Some(ref pub_socket) = zmq_pub {
                        pub_socket.send_ble(&p, gps_fix);
                    }
                }
            }
            return;
        }
    };

    let freq = burst.freq;
    let burst_ts = burst.timestamp.clone();
    let rssi = burst.rssi_db as i32;
    let noise = burst.noise_db as i32;

    // Try Classic BT first
    if let Some(mut bt_pkt) = btbb::detect(
        &fsk_result.bits,
        freq,
        rssi,
        noise,
        burst_ts.clone(),
        syndrome_map,
    ) {
        let demod_ts = channel_samples_after(&burst_ts, burst.edr_lead_samples);
        bt_pkt.timestamp = bt_sync_timestamp(&demod_ts, &fsk_result, bt_pkt.sync_offset);
        // Recover UAP (and full BD_ADDR from FHS) across packets/channels.
        let mut announce = btbb::enrich(&mut bt_pkt, bt_tracker);
        let offset_hz = (burst.freq as f64 - center_freq_mhz as f64) * 1_000_000.0;
        let edr_attempt = try_enrich_edr(
            burst,
            &fsk_result,
            &mut bt_pkt,
            bt_tracker,
            wideband,
            offset_hz,
            raw_sample_rate,
            burst.edr_lead_samples,
        );
        stats.record_edr(&edr_attempt);
        if edr_attempt.enriched {
            announce |= bt_tracker.mark_announced(bt_pkt.lap);
        }
        if announce {
            log_bt_address(&bt_pkt);
        }
        stats.total_bt += 1;
        if bt_pkt.crc_ok {
            stats.total_crc += 1;
            stats.valid_crc += 1;
        }
        if let Some(ref mut writer) = pcap_writer {
            if let Err(e) = writer.write_bt(&bt_pkt, gps_fix) {
                if stats.pcap_errors == 0 {
                    eprintln!("PCAP write error: {}", e);
                }
                stats.pcap_errors += 1;
            }
        }
        #[cfg(feature = "zmq")]
        if let Some(ref pub_socket) = zmq_pub {
            pub_socket.send_bt(&bt_pkt, gps_fix);
        }
        return;
    }

    // BLE channels are spaced every 2 MHz. Odd-MHz bins are Classic-only.
    if freq & 1 != 0 {
        return;
    }

    // For long bursts (> 2000 samples), try coded FIRST since coded packets
    // are always long (min ~2700 at S=2, ~6800+ at S=8). The coded preamble
    // check is cheap and highly distinctive (80 symbols of 00111100).
    let mut pkt = None;
    let burst_len = fsk_result.demod.len();

    if burst_len > 2000 {
        stats.coded_attempts += 1;
        stats.coded_fsk_ok += 1;
        pkt = ble::ble_coded_burst(
            &fsk_result.demod,
            freq,
            burst_ts.clone(),
            2, // SPS=2
            check_crc,
            |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
        );
    }

    // Try BLE LE 1M preamble-first detection
    if pkt.is_none() {
        pkt = ble::ble_burst(&fsk_result.bits, freq, burst_ts.clone(), check_crc, |aa| {
            conn_table.crc_init_for_aa(aa, burst_ts.tv_sec)
        });
    }

    // Fall back to LE 1M AA correlator
    if pkt.is_none() {
        pkt = aa_correlator.correlate(&fsk_result.demod, freq, burst_ts.clone(), check_crc);
    }

    // Try LE 2M: reslice the demod at SPS=1 and try preamble-first
    if pkt.is_none() {
        let bits_2m = fsk::reslice(&fsk_result.demod, fsk_result.silence, 1);
        pkt = ble::ble_burst_2m(&bits_2m, freq, burst_ts.clone(), check_crc, |aa| {
            conn_table.crc_init_for_aa(aa, burst_ts.tv_sec)
        });

        // Fall back to LE 2M AA correlator
        if pkt.is_none() {
            pkt =
                aa_correlator_2m.correlate_2m(&fsk_result.demod, freq, burst_ts.clone(), check_crc);
        }
    }

    // Also try coded on shorter bursts (S=2 min is ~2000 samples)
    if pkt.is_none() && burst_len <= 2000 {
        stats.coded_attempts += 1;
        pkt = ble::ble_coded_burst(
            &fsk_result.demod,
            freq,
            burst_ts.clone(),
            2, // SPS=2
            check_crc,
            |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
        );
    }

    if let Some(mut p) = pkt {
        p.rssi_db = rssi;
        p.noise_db = noise;

        if p.aa == ble::BLE_ADV_AA && p.crc_valid {
            conn_table.parse_connect_ind(&p, burst_ts.tv_sec);
        }

        // Parse SMP from data channel packets (non-advertising AA with valid CRC)
        if p.aa != ble::BLE_ADV_AA && p.crc_valid && p.is_data {
            if let Some((l2cap_cid, l2cap_payload)) = bd_protocol::smp::extract_l2cap(&p.data) {
                // Direction: assume central is the initiator for now
                // (proper direction tracking requires connection role from CONNECT_IND)
                let from_central = true;
                let smp_events =
                    smp_parser.parse_l2cap(p.aa, l2cap_cid, l2cap_payload, from_central);
                for event in &smp_events {
                    // Log to stderr
                    match event {
                        bd_protocol::smp::SmpEvent::FeaturesExchanged {
                            aa,
                            method,
                            security,
                            ..
                        } => {
                            eprintln!(
                                "SMP: connection 0x{:08X} pairing {:?} ({:?})",
                                aa, method, security
                            );
                        }
                        bd_protocol::smp::SmpEvent::WeakPairing { aa, reason } => {
                            eprintln!("SMP WARNING: connection 0x{:08X}: {}", aa, reason);
                        }
                        bd_protocol::smp::SmpEvent::LtkDistributed { aa, .. } => {
                            eprintln!("SMP: connection 0x{:08X} LTK captured", aa);
                        }
                        bd_protocol::smp::SmpEvent::IrkDistributed { aa, .. } => {
                            eprintln!("SMP: connection 0x{:08X} IRK captured", aa);
                        }
                        bd_protocol::smp::SmpEvent::PairingFailed { aa, reason } => {
                            eprintln!(
                                "SMP: connection 0x{:08X} pairing failed (reason {})",
                                aa, reason
                            );
                        }
                        _ => {}
                    }
                    // Stream to dashboard via ZMQ
                    #[cfg(feature = "zmq")]
                    if let Some(ref pub_socket) = zmq_pub {
                        let json = format_smp_event(event);
                        pub_socket.send_smp(&json);
                    }
                }
            }
        }

        // Debug: show ADV_EXT_IND packets with AuxPtr to find coded secondary channel
        if let Some(ref eh) = p.ext_header {
            if let Some(ref aux) = eh.aux_ptr {
                let phy_str = match aux.phy {
                    ble::BlePhy::Phy1M => "1M",
                    ble::BlePhy::Phy2M => "2M",
                    ble::BlePhy::PhyCoded => "Coded",
                };
                let ch_freq = 2402 + (aux.channel as u32) * 2;
                eprintln!(
                    "ADV_EXT_IND: freq={} rssi={} -> AuxPtr ch={} ({}MHz) phy={} offset={}us",
                    freq, rssi, aux.channel, ch_freq, phy_str, aux.offset_usec
                );
            }
        }

        if p.crc_checked {
            stats.total_crc += 1;
            if p.crc_valid {
                stats.valid_crc += 1;
            }
        }

        match p.phy {
            ble::BlePhy::Phy2M => stats.total_ble_2m += 1,
            ble::BlePhy::PhyCoded => stats.total_ble_coded += 1,
            _ => {}
        }

        stats.total_ble += 1;
        if let Some(ref mut writer) = pcap_writer {
            if let Err(e) = writer.write_ble(&p, gps_fix) {
                if stats.pcap_errors == 0 {
                    eprintln!("PCAP write error: {}", e);
                }
                stats.pcap_errors += 1;
            }
        }
        #[cfg(feature = "zmq")]
        if let Some(ref pub_socket) = zmq_pub {
            pub_socket.send_ble(&p, gps_fix);
        }
    }
}

struct PipelineStats {
    total_ble: u64,
    total_ble_2m: u64,
    total_ble_coded: u64,
    total_bt: u64,
    total_crc: u64,
    valid_crc: u64,
    total_bursts: u64,
    edr_attempts: u64,
    edr_syncs: u64,
    edr_crc_matches: u64,
    edr_best_sync_score: Option<f32>,
    /// Debug: bursts that reached the coded decoder (all prior decoders returned None)
    coded_attempts: u64,
    /// Debug: FSK demod failures >= 1500 samples
    fsk_reject_long: u64,
    /// Debug: coded attempts from FSK-success path
    coded_fsk_ok: u64,
    /// Debug: longest burst seen (samples)
    max_burst_len: usize,
    /// Debug: burst length histogram
    burst_lt200: u64,
    burst_200_1k: u64,
    burst_1k_5k: u64,
    burst_5k_50k: u64,
    burst_50k_plus: u64,
    pcap_errors: u64,
}

impl PipelineStats {
    fn new() -> Self {
        Self {
            total_ble: 0,
            total_ble_2m: 0,
            total_ble_coded: 0,
            total_bt: 0,
            total_crc: 0,
            valid_crc: 0,
            total_bursts: 0,
            edr_attempts: 0,
            edr_syncs: 0,
            edr_crc_matches: 0,
            edr_best_sync_score: None,
            coded_attempts: 0,
            fsk_reject_long: 0,
            coded_fsk_ok: 0,
            max_burst_len: 0,
            burst_lt200: 0,
            burst_200_1k: 0,
            burst_1k_5k: 0,
            burst_5k_50k: 0,
            burst_50k_plus: 0,
            pcap_errors: 0,
        }
    }

    fn crc_pct(&self) -> f64 {
        if self.total_crc > 0 {
            (self.valid_crc as f64 / self.total_crc as f64) * 100.0
        } else {
            0.0
        }
    }

    fn record_edr(&mut self, attempt: &EdrAttempt) {
        self.edr_attempts += u64::from(attempt.candidate);
        self.edr_syncs += u64::from(attempt.synchronized);
        self.edr_crc_matches += u64::from(attempt.crc_match);
        if let Some(score) = attempt.best_sync_score {
            if self.edr_best_sync_score.is_none_or(|best| score < best) {
                self.edr_best_sync_score = Some(score);
            }
        }
    }
}

/// Message sent from main thread to burst worker threads.
struct BatchMsg {
    data: Arc<Vec<f32>>,
    /// Interleaved raw SC16 IQ covering the same time span as `data`.
    /// Present only for the opt-in EDR wideband path.
    raw_iq: Option<Arc<Vec<i16>>>,
    batch_steps: usize,
    ts: Timespec,
}

struct BurstMsg {
    burst: bd_dsp::burst::Burst,
    wideband: Option<Vec<Complex32>>,
}

struct RawBatch {
    iq: Arc<Vec<i16>>,
    start_sample: u128,
}

struct RawBatchCache {
    batches: std::collections::VecDeque<RawBatch>,
    sample_rate: u32,
}

impl RawBatchCache {
    fn new(sample_rate: u32) -> Self {
        Self {
            batches: std::collections::VecDeque::new(),
            sample_rate,
        }
    }

    fn sample_index(&self, ts: &Timespec) -> u128 {
        ts.tv_sec as u128 * self.sample_rate as u128
            + ts.tv_nsec as u128 * self.sample_rate as u128 / 1_000_000_000
    }

    fn push(&mut self, msg: &BatchMsg) {
        let Some(iq) = msg.raw_iq.clone() else {
            return;
        };
        self.batches.push_back(RawBatch {
            iq,
            start_sample: self.sample_index(&msg.ts),
        });
        while self.batches.len() > 8 {
            self.batches.pop_front();
        }
    }

    fn extract(&self, start: &Timespec, duration_us: u32) -> Option<Vec<Complex32>> {
        let wanted_start = self.sample_index(start);
        let wanted_len = self.sample_rate as usize * duration_us as usize / 1_000_000;
        let wanted_end = wanted_start + wanted_len as u128;
        let mut output = Vec::with_capacity(wanted_len);
        let mut next_sample = wanted_start;

        for batch in &self.batches {
            let batch_len = batch.iq.len() / 2;
            let batch_end = batch.start_sample + batch_len as u128;
            if batch_end <= next_sample || batch.start_sample >= wanted_end {
                continue;
            }
            if batch.start_sample > next_sample {
                return None;
            }
            let first = (next_sample - batch.start_sample) as usize;
            let last = ((wanted_end.min(batch_end)) - batch.start_sample) as usize;
            for index in first..last {
                output.push(Complex32::new(
                    batch.iq[2 * index] as f32,
                    batch.iq[2 * index + 1] as f32,
                ));
            }
            next_sample = batch.start_sample + last as u128;
            if next_sample >= wanted_end {
                return Some(output);
            }
        }
        None
    }
}

/// Channel assignment for a single burst worker.
#[derive(Clone)]
struct ChannelAssignment {
    ch_idx: usize,
    fft_bin: usize,
}

/// Per-channel adaptive gain. Tracks rolling chunk-RMS and snaps the per-bin
/// scale so each channel's noise floor sits near `TARGET_NOISE_RMS`. This
/// equalizes the AGC operating point across all active channels even when
/// WiFi raises some bins by 30+ dB -- without per-bin equalization, the
/// shared squelch threshold is biased by the WiFi-noisy channels and the
/// AGC takes longer to converge on burst start (corrupting early symbols).
struct ChannelGain {
    chunk_sum_sq: f32,
    chunk_count: u32,
    /// Rolling buffer of chunk RMSes; oldest entry overwritten each cycle.
    history: [f32; CHANNEL_GAIN_HISTORY],
    history_idx: usize,
    history_filled: bool,
    chunks_since_update: u32,
    scale: f32,
    /// Upper clamp on the per-bin scale. Different in Full vs halfband modes.
    max_scale: f32,
    /// Latest p25 estimate (0.0 until first window completes).
    last_p25: f32,
    /// Target the bin scales toward. Set externally each resync to the
    /// median p25 across the worker's channels, so the gain block tracks
    /// the actual noise environment instead of a fixed reference.
    target_p25: f32,
}

const CHANNEL_GAIN_CHUNK: u32 = 256; // ~256 µs per chunk at 1 Msps/ch
const CHANNEL_GAIN_HISTORY: usize = 64; // 64 chunks ≈ 16 ms total window
const CHANNEL_GAIN_UPDATE_CHUNKS: u32 = 64; // recompute scale every full window
const CHANNEL_GAIN_TARGET: f32 = 0.01; // post-scale RMS target (per channel)
const CHANNEL_GAIN_INIT: f32 = 1.0; // pass-through until we have data
                                    // Clamp range for the per-channel adaptive gain. The block is enabled
                                    // only in decim>1 (halfband+) modes, where some bins receive WiFi-loud
                                    // energy and others are filter-rejected; attenuate-only equalization
                                    // pulls the loud bins down to the cross-channel median so the AGC sees
                                    // a uniform operating point. Amplification (max>1) was tried for
                                    // decim=1 mode but was unstable across RF environments, so the gain
                                    // block is disabled there entirely.
const CHANNEL_GAIN_MIN: f32 = 0.1; // attenuate by up to 20 dB
const CHANNEL_GAIN_MAX_DECIM: f32 = 1.0; // never amplify

impl ChannelGain {
    fn new(max_scale: f32) -> Self {
        Self {
            chunk_sum_sq: 0.0,
            chunk_count: 0,
            history: [0.0; CHANNEL_GAIN_HISTORY],
            history_idx: 0,
            history_filled: false,
            chunks_since_update: 0,
            scale: CHANNEL_GAIN_INIT,
            max_scale,
            last_p25: 0.0,
            // Bootstrap to a fixed reference until the worker has enough
            // channels with p25 data to compute a median.
            target_p25: CHANNEL_GAIN_TARGET,
        }
    }

    fn last_p25(&self) -> f32 {
        self.last_p25
    }

    /// Set the cross-channel reference. Worker calls this every ~tens-of-ms
    /// with the median p25 across its assigned channels. The new scale is
    /// applied on the next sample.
    fn set_target_p25(&mut self, target: f32) {
        self.target_p25 = target;
        self.recompute_scale();
    }

    fn recompute_scale(&mut self) {
        if self.last_p25 > 1e-9 {
            self.scale = (self.target_p25 / self.last_p25).clamp(CHANNEL_GAIN_MIN, self.max_scale);
        }
    }

    /// Apply current scale to `sample` and update the rolling RMS estimator.
    /// The scale only changes once per full window (~16 ms) to avoid step
    /// changes mid-burst.
    fn process(&mut self, sample: num_complex::Complex32) -> num_complex::Complex32 {
        let mag_sq = sample.norm_sqr();
        self.chunk_sum_sq += mag_sq;
        self.chunk_count += 1;

        if self.chunk_count >= CHANNEL_GAIN_CHUNK {
            let chunk_rms = (self.chunk_sum_sq / self.chunk_count as f32).sqrt();
            self.history[self.history_idx] = chunk_rms;
            self.history_idx += 1;
            if self.history_idx >= CHANNEL_GAIN_HISTORY {
                self.history_idx = 0;
                self.history_filled = true;
            }
            self.chunk_sum_sq = 0.0;
            self.chunk_count = 0;
            self.chunks_since_update += 1;

            if self.chunks_since_update >= CHANNEL_GAIN_UPDATE_CHUNKS && self.history_filled {
                // 25th percentile of chunk RMSes ≈ noise floor when bursts
                // are <75% duty cycle (always true for BLE; usually true for
                // WiFi channels too because WiFi has gaps between frames).
                let mut sorted = self.history;
                sorted
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.last_p25 = sorted[CHANNEL_GAIN_HISTORY / 4];
                self.recompute_scale();
                self.chunks_since_update = 0;
            }
        }

        sample * self.scale
    }
}

/// Spawn parallel burst-catching worker threads + a decode thread.
///
/// Returns:
/// - `batch_txs`: one sender per worker (main thread broadcasts batches to all)
/// - `worker_handles`: join handles for all worker threads
/// - `decode_handle`: join handle for the decode thread
///
/// The decode thread handles FSK demod, BLE/BT decode, PCAP/ZMQ output, and stats.
#[allow(clippy::too_many_arguments)]
fn spawn_parallel_pipeline(
    num_channels: usize,
    center_freq_mhz: u32,
    raw_sample_rate: u32,
    edr_wideband_enabled: bool,
    fft_scale: f32,
    channel_gain_max: f32,
    channel_gain_enabled: bool,
    mut burst_catchers: Vec<Option<BurstCatcher>>,
    fsk: FskDemod,
    aa_correlator: AaCorrelator,
    aa_correlator_2m: AaCorrelator,
    syndrome_map: SyndromeMap,
    conn_table: ConnectionTable,
    classic_uaps: Vec<(u32, u8)>,
    pcap_writer: Option<PcapWriter<BufWriter<File>>>,
    burst_writer: Option<FileBurstWriter>,
    check_crc: bool,
    print_stats: bool,
    overflow_count: Arc<std::sync::atomic::AtomicU64>,
    squelch_pending: Arc<AtomicI32>,
    #[cfg(feature = "zmq")] zmq_config: Option<(String, Option<String>, Option<String>)>,
    #[cfg(feature = "zmq")] hb_state: Option<Arc<Mutex<bd_output::control::HeartbeatState>>>,
    #[cfg(feature = "gps")] gps_client: Option<bd_output::gps::GpsClient>,
) -> (
    Vec<channel::Sender<BatchMsg>>,
    Vec<std::thread::JoinHandle<()>>,
    std::thread::JoinHandle<()>,
) {
    // Collect active channels
    let active: Vec<ChannelAssignment> = burst_catchers
        .iter()
        .enumerate()
        .filter_map(|(fft_bin, catcher)| {
            if catcher.is_some() {
                Some(ChannelAssignment {
                    ch_idx: fft_bin,
                    fft_bin,
                })
            } else {
                None
            }
        })
        .collect();

    let hw_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    // Scale workers with channel count: allow up to hw_threads-2 (reserve for
    // PFB and SDR recv threads), capped by the number of active channels.
    let n_workers = active.len().min(hw_threads.saturating_sub(2).max(4)).max(1);

    // Burst output channel: all workers send here, decode thread receives
    let (burst_tx, burst_rx) = channel::bounded::<BurstMsg>(512);

    let mut batch_txs = Vec::with_capacity(n_workers);
    let mut worker_handles = Vec::with_capacity(n_workers);

    let chunk_size = (active.len() + n_workers - 1) / n_workers;

    for (worker_id, chunk) in active.chunks(chunk_size).enumerate() {
        let (batch_tx, batch_rx) = channel::bounded::<BatchMsg>(4);
        batch_txs.push(batch_tx);

        let channels: Vec<ChannelAssignment> = chunk.to_vec();
        let mut catchers: Vec<BurstCatcher> = channels
            .iter()
            .map(|a| burst_catchers[a.ch_idx].take().unwrap())
            .collect();
        let mut gains: Vec<ChannelGain> = (0..channels.len())
            .map(|_| ChannelGain::new(channel_gain_max))
            .collect();

        let burst_tx = burst_tx.clone();
        let num_ch = num_channels;
        let scale = fft_scale;
        let sq_pending = squelch_pending.clone();

        let handle = std::thread::Builder::new()
            .name(format!("burst-{}", worker_id))
            .spawn(move || {
                let mut current_squelch = i32::MIN;
                let mut steps_since_resync: usize = 0;
                let mut raw_cache = RawBatchCache::new(raw_sample_rate);
                // Resync the per-channel gain target every ~64 ms at 1 Msps/ch.
                // Long enough that p25 estimates have all updated at least
                // once (each updates every ~16 ms), short enough to track a
                // changing RF environment.
                const RESYNC_STEPS: usize = 65536;
                for msg in batch_rx.iter() {
                    if edr_wideband_enabled {
                        raw_cache.push(&msg);
                    }
                    // Check for squelch update
                    let sq = sq_pending.load(Ordering::Relaxed);
                    if sq != current_squelch && sq != i32::MIN {
                        let threshold = sq as f32 / 10.0;
                        for c in catchers.iter_mut() {
                            c.set_squelch(threshold);
                        }
                        current_squelch = sq;
                    }

                    for (i, assign) in channels.iter().enumerate() {
                        let catcher = &mut catchers[i];
                        let gain = &mut gains[i];
                        for t in 0..msg.batch_steps {
                            let base = t * num_ch * 2;
                            let idx = base + assign.fft_bin * 2;
                            let raw = num_complex::Complex32::new(
                                msg.data[idx] * scale,
                                msg.data[idx + 1] * scale,
                            );
                            let sample = if channel_gain_enabled {
                                gain.process(raw)
                            } else {
                                raw
                            };
                            if let Some(burst) = catcher.execute_at(sample, &msg.ts, t) {
                                // Grab the raw wideband window for EDR-extended
                                // bursts and any multi-slot-length burst: a
                                // clipped EDR packet often detects as a Basic Rate
                                // multi-slot type, and only the wideband path can
                                // recover its DPSK payload. Short bursts skip it.
                                let wideband = if edr_wideband_enabled
                                    && (burst.edr_extended || burst.samples.len() >= 1000)
                                {
                                    raw_cache.extract(&burst.timestamp, 3_100)
                                } else {
                                    None
                                };
                                let _ = burst_tx.send(BurstMsg { burst, wideband });
                            }
                        }
                        // Emit scan bursts for advertising channels
                        if let Some(scan_burst) = catcher.take_scan_burst() {
                            let _ = burst_tx.send(BurstMsg {
                                burst: scan_burst,
                                wideband: None,
                            });
                        }
                    }

                    // Cross-channel resync: each bin scales toward
                    // max(fixed reference, median of worker's channels). Only
                    // runs when the gain block is enabled (decim>1 mode).
                    if !channel_gain_enabled {
                        continue;
                    }
                    steps_since_resync += msg.batch_steps;
                    if steps_since_resync >= RESYNC_STEPS {
                        steps_since_resync = 0;
                        let mut p25s: Vec<f32> = gains
                            .iter()
                            .map(|g| g.last_p25())
                            .filter(|&p| p > 1e-9)
                            .collect();
                        if p25s.len() >= 2 {
                            p25s.sort_unstable_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let median = p25s[p25s.len() / 2];
                            let target = median.max(CHANNEL_GAIN_TARGET);
                            for g in gains.iter_mut() {
                                g.set_target_p25(target);
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn burst worker");

        worker_handles.push(handle);
    }

    // Drop the original burst_tx so decode thread terminates when all workers finish
    drop(burst_tx);

    eprintln!(
        "pipeline: {} burst workers, {} channels",
        n_workers,
        active.len()
    );

    // Decode thread: FSK demod + BLE/BT protocol decode + output
    let decode_handle = {
        let overflow_proc = overflow_count;
        let mut fsk = fsk;
        let aa_correlator = aa_correlator;
        let aa_correlator_2m = aa_correlator_2m;
        let syndrome_map = syndrome_map;
        let mut conn_table = conn_table;
        let mut bt_tracker = btbb::PiconetTracker::new();
        seed_classic_uaps(&mut bt_tracker, &classic_uaps);
        let mut smp_parser = bd_protocol::smp::SmpParser::new();
        let mut pcap_writer = pcap_writer;
        let mut burst_writer = burst_writer;

        std::thread::Builder::new()
            .name("decode".to_string())
            .spawn(move || {
                let mut stats = PipelineStats::new();
                let stats_start = Instant::now();
                let mut last_stats = Instant::now();

                // Create ZMQ publisher inside this thread (zmq::Socket is !Send)
                #[cfg(feature = "zmq")]
                let zmq_pub: Option<bd_output::zmq_pub::ZmqPublisher> =
                    zmq_config.and_then(|(ep, sid, curve_kf)| {
                        match bd_output::zmq_pub::ZmqPublisher::new(
                            &ep,
                            sid.as_deref(),
                            curve_kf.as_deref(),
                        ) {
                            Ok(p) => Some(p),
                            Err(e) => {
                                eprintln!("ZMQ PUB: {}", e);
                                None
                            }
                        }
                    });

                #[cfg(feature = "gps")]
                let gps_client = gps_client;

                for burst_msg in burst_rx.iter() {
                    let burst = burst_msg.burst;
                    record_burst(&mut burst_writer, &burst);
                    #[cfg(feature = "gps")]
                    let gps_fix = gps_client.as_ref().map(|c| c.get_fix());
                    #[cfg(not(feature = "gps"))]
                    let gps_fix: Option<bd_output::pcap::GpsFix> = None;
                    let gps_ref = gps_fix.as_ref().filter(|f| f.valid);

                    process_burst(
                        &burst,
                        burst_msg.wideband.as_deref(),
                        center_freq_mhz,
                        raw_sample_rate,
                        &mut fsk,
                        &aa_correlator,
                        &aa_correlator_2m,
                        &syndrome_map,
                        &mut bt_tracker,
                        &mut conn_table,
                        &mut smp_parser,
                        &mut pcap_writer,
                        #[cfg(feature = "zmq")]
                        &zmq_pub,
                        gps_ref,
                        check_crc,
                        &mut stats,
                    );

                    if last_stats.elapsed().as_secs() >= 5 {
                        // Always update heartbeat for C2 (even without --stats)
                        #[cfg(feature = "zmq")]
                        if let Some(ref hb) = hb_state {
                            let elapsed = stats_start.elapsed().as_secs_f64();
                            if let Ok(mut s) = hb.lock() {
                                s.total_pkts = stats.total_ble + stats.total_bt;
                                s.pkt_rate = (stats.total_ble + stats.total_bt) as f64 / elapsed;
                                s.crc_pct = stats.crc_pct();
                            }
                        }

                        if !print_stats {
                            last_stats = Instant::now();
                        }
                    }
                    if print_stats && last_stats.elapsed().as_secs() >= 5 {
                        let elapsed = stats_start.elapsed().as_secs_f64();
                        let conns = conn_table.count();
                        let overflows = overflow_proc.load(Ordering::Relaxed);
                        let phy_str = if stats.total_ble_2m > 0 || stats.total_ble_coded > 0 {
                            format!(" (2M:{} coded:{})", stats.total_ble_2m, stats.total_ble_coded)
                        } else {
                            String::new()
                        };
                        eprint!(
                            "[{:.1}s] BLE: {}{} BT: {} bursts: {} CRC: {:.1}% ({}/{}) conns: {} overflow: {} EDR:{}/{}/{} best:{:.3} coded_try:{} coded_ok:{} fsk_rej:{} max_burst:{} lens:<200:{} 200-1k:{} 1k-5k:{} 5k-50k:{} 50k+:{}\n",
                            elapsed,
                            stats.total_ble,
                            phy_str,
                            stats.total_bt,
                            stats.total_bursts,
                            stats.crc_pct(),
                            stats.valid_crc,
                            stats.total_crc,
                            conns,
                            overflows,
                            stats.edr_attempts,
                            stats.edr_syncs,
                            stats.edr_crc_matches,
                            stats.edr_best_sync_score.unwrap_or(f32::NAN),
                            stats.coded_attempts,
                            stats.coded_fsk_ok,
                            stats.fsk_reject_long,
                            stats.max_burst_len,
                            stats.burst_lt200,
                            stats.burst_200_1k,
                            stats.burst_1k_5k,
                            stats.burst_5k_50k,
                            stats.burst_50k_plus,
                        );

                        last_stats = Instant::now();
                    }
                }

                if print_stats {
                    let elapsed = stats_start.elapsed().as_secs_f64();
                    let overflows = overflow_proc.load(Ordering::Relaxed);
                    let phy_str = if stats.total_ble_2m > 0 || stats.total_ble_coded > 0 {
                        format!(" (2M:{} coded:{})", stats.total_ble_2m, stats.total_ble_coded)
                    } else {
                        String::new()
                    };
                    eprintln!(
                        "done ({:.1}s): BLE: {}{} BT: {} bursts: {} CRC: {:.1}% ({}/{}) overflow: {} EDR: try={} sync={} crc={} best={:.3}",
                        elapsed,
                        stats.total_ble,
                        phy_str,
                        stats.total_bt,
                        stats.total_bursts,
                        stats.crc_pct(),
                        stats.valid_crc,
                        stats.total_crc,
                        overflows,
                        stats.edr_attempts,
                        stats.edr_syncs,
                        stats.edr_crc_matches,
                        stats.edr_best_sync_score.unwrap_or(f32::NAN),
                    );
                }
            })
            .expect("failed to spawn decode thread")
    };

    (batch_txs, worker_handles, decode_handle)
}

/// Broadcast a batch to all worker threads.
#[inline]
/// Format an SMP event as a JSON string for ZMQ streaming.
fn format_smp_event(event: &bd_protocol::smp::SmpEvent) -> String {
    use bd_protocol::smp::SmpEvent;
    match event {
        SmpEvent::PairingStarted { aa } => {
            format!(r#"{{"event":"pairing_started","aa":"0x{:08X}"}}"#, aa)
        }
        SmpEvent::FeaturesExchanged {
            aa,
            method,
            security,
            initiator,
            responder,
        } => {
            format!(
                r#"{{"event":"features_exchanged","aa":"0x{:08X}","method":"{:?}","security":"{:?}","init_io":"{}","resp_io":"{}"}}"#,
                aa,
                method,
                security,
                initiator.io_capability_str(),
                responder.io_capability_str()
            )
        }
        SmpEvent::WeakPairing { aa, reason } => {
            format!(
                r#"{{"event":"weak_pairing","aa":"0x{:08X}","reason":"{}"}}"#,
                aa,
                reason.replace('"', "'")
            )
        }
        SmpEvent::LtkDistributed { aa, ltk } => {
            let hex: String = ltk.iter().map(|b| format!("{:02x}", b)).collect();
            format!(
                r#"{{"event":"ltk_distributed","aa":"0x{:08X}","ltk":"{}"}}"#,
                aa, hex
            )
        }
        SmpEvent::IrkDistributed { aa, irk } => {
            let hex: String = irk.iter().map(|b| format!("{:02x}", b)).collect();
            format!(
                r#"{{"event":"irk_distributed","aa":"0x{:08X}","irk":"{}"}}"#,
                aa, hex
            )
        }
        SmpEvent::CsrkDistributed { aa, csrk } => {
            let hex: String = csrk.iter().map(|b| format!("{:02x}", b)).collect();
            format!(
                r#"{{"event":"csrk_distributed","aa":"0x{:08X}","csrk":"{}"}}"#,
                aa, hex
            )
        }
        SmpEvent::IdentityAddress {
            aa,
            addr_type,
            addr,
        } => {
            let mac: String = addr
                .iter()
                .rev()
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(":");
            format!(
                r#"{{"event":"identity_address","aa":"0x{:08X}","addr_type":{},"addr":"{}"}}"#,
                aa, addr_type, mac
            )
        }
        SmpEvent::PairingFailed { aa, reason } => {
            format!(
                r#"{{"event":"pairing_failed","aa":"0x{:08X}","reason":{}}}"#,
                aa, reason
            )
        }
        SmpEvent::PairingConfirm {
            aa, from_initiator, ..
        } => {
            format!(
                r#"{{"event":"pairing_confirm","aa":"0x{:08X}","from_initiator":{}}}"#,
                aa, from_initiator
            )
        }
        SmpEvent::PairingRandom {
            aa, from_initiator, ..
        } => {
            format!(
                r#"{{"event":"pairing_random","aa":"0x{:08X}","from_initiator":{}}}"#,
                aa, from_initiator
            )
        }
        SmpEvent::PublicKey { aa, from_initiator } => {
            format!(
                r#"{{"event":"public_key","aa":"0x{:08X}","from_initiator":{}}}"#,
                aa, from_initiator
            )
        }
    }
}

fn broadcast_batch(
    txs: &[channel::Sender<BatchMsg>],
    data: Vec<f32>,
    raw_iq: Option<Vec<i16>>,
    batch_steps: usize,
    ts: &Timespec,
) {
    let arc = Arc::new(data);
    let raw_iq = raw_iq.map(Arc::new);
    for tx in txs {
        let _ = tx.send(BatchMsg {
            data: arc.clone(),
            raw_iq: raw_iq.clone(),
            batch_steps,
            ts: ts.clone(),
        });
    }
}

fn channel_samples_after(start: &Timespec, samples: usize) -> Timespec {
    const CHANNEL_SAMPLE_PERIOD_NS: u64 = 500;
    let offset_ns = samples as u64 * CHANNEL_SAMPLE_PERIOD_NS;
    let total_ns = start.tv_nsec + offset_ns;
    Timespec {
        tv_sec: start.tv_sec + total_ns / 1_000_000_000,
        tv_nsec: total_ns % 1_000_000_000,
    }
}

fn initial_batch_timestamp(steps: usize) -> Timespec {
    const CHANNEL_SAMPLE_PERIOD_NS: u64 = 500;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let now_ns = now.as_secs() as u128 * 1_000_000_000 + now.subsec_nanos() as u128;
    let duration_ns = steps.saturating_sub(1) as u128 * CHANNEL_SAMPLE_PERIOD_NS as u128;
    let start_ns = now_ns.saturating_sub(duration_ns);
    Timespec {
        tv_sec: (start_ns / 1_000_000_000) as u64,
        tv_nsec: (start_ns % 1_000_000_000) as u64,
    }
}

/// SDR backend type, detected from interface string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SdrType {
    Vita49,
    Usrp,
    HackRf,
    BladeRf,
    SoapySdr,
    Aaronia,
    Rfnm,
    Sidekiq,
}

impl std::fmt::Display for SdrType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SdrType::Vita49 => write!(f, "vita49"),
            SdrType::Usrp => write!(f, "usrp"),
            SdrType::HackRf => write!(f, "hackrf"),
            SdrType::BladeRf => write!(f, "bladerf"),
            SdrType::SoapySdr => write!(f, "soapysdr"),
            SdrType::Aaronia => write!(f, "aaronia"),
            SdrType::Rfnm => write!(f, "rfnm"),
            SdrType::Sidekiq => write!(f, "sidekiq"),
        }
    }
}

/// Detect SDR backend type from interface string.
fn detect_sdr_type(iface: &str) -> SdrType {
    if iface.starts_with("vita49") {
        SdrType::Vita49
    } else if iface.starts_with("usrp") {
        SdrType::Usrp
    } else if iface.starts_with("hackrf") {
        SdrType::HackRf
    } else if iface.starts_with("bladerf") {
        SdrType::BladeRf
    } else if iface.starts_with("soapy") {
        SdrType::SoapySdr
    } else if iface.starts_with("aaronia") {
        SdrType::Aaronia
    } else if iface.starts_with("rfnm") {
        SdrType::Rfnm
    } else if iface.starts_with("sidekiq") {
        SdrType::Sidekiq
    } else {
        SdrType::Usrp // default
    }
}

/// Abstraction over SDR handle backends for the recv_into() path.
enum SdrHandle {
    #[cfg(feature = "usrp")]
    Usrp(bd_sdr::usrp::UsrpHandle),
    #[cfg(feature = "hackrf")]
    HackRf(bd_sdr::hackrf::HackrfHandle),
    #[cfg(feature = "bladerf")]
    BladeRf(bd_sdr::bladerf::BladerfHandle),
    #[cfg(feature = "soapysdr")]
    Soapy(bd_sdr::soapysdr::SoapyHandle),
    #[cfg(feature = "aaronia")]
    Aaronia(bd_sdr::aaronia::AaroniaHandle),
    #[cfg(feature = "rfnm")]
    Rfnm(bd_sdr::rfnm::RfnmHandle),
    #[cfg(feature = "sidekiq")]
    Sidekiq(bd_sdr::sidekiq::SidekiqHandle),
    Vita49(bd_sdr::vita49::Vita49Handle),
}

// Safety: SDR C library handles are thread-safe (recv from one thread is fine).
// The raw pointer inside UsrpHandle/etc. is an opaque C handle that supports this.
unsafe impl Send for SdrHandle {}

impl SdrHandle {
    fn recv_into_i16(&mut self, buf: &mut [i16]) -> usize {
        match self {
            #[cfg(feature = "usrp")]
            SdrHandle::Usrp(h) => h.recv_into_i16(buf),
            #[cfg(feature = "hackrf")]
            SdrHandle::HackRf(h) => h.recv_into_i16(buf),
            #[cfg(feature = "bladerf")]
            SdrHandle::BladeRf(h) => h.recv_into_i16(buf),
            #[cfg(feature = "soapysdr")]
            SdrHandle::Soapy(h) => h.recv_into_i16(buf),
            #[cfg(feature = "aaronia")]
            SdrHandle::Aaronia(h) => h.recv_into_i16(buf),
            #[cfg(feature = "rfnm")]
            SdrHandle::Rfnm(h) => h.recv_into_i16(buf),
            #[cfg(feature = "sidekiq")]
            SdrHandle::Sidekiq(h) => h.recv_into_i16(buf),
            SdrHandle::Vita49(h) => h.recv_into_i16(buf),
        }
    }

    fn max_samps(&self) -> usize {
        match self {
            #[cfg(feature = "usrp")]
            SdrHandle::Usrp(h) => h.max_samps(),
            #[cfg(feature = "hackrf")]
            SdrHandle::HackRf(h) => h.max_samps(),
            #[cfg(feature = "bladerf")]
            SdrHandle::BladeRf(h) => h.max_samps(),
            #[cfg(feature = "soapysdr")]
            SdrHandle::Soapy(h) => h.max_samps(),
            #[cfg(feature = "aaronia")]
            SdrHandle::Aaronia(h) => h.max_samps(),
            #[cfg(feature = "rfnm")]
            SdrHandle::Rfnm(h) => h.max_samps(),
            #[cfg(feature = "sidekiq")]
            SdrHandle::Sidekiq(h) => h.max_samps(),
            SdrHandle::Vita49(h) => h.max_samps(),
        }
    }

    fn overflow_count(&self) -> u64 {
        match self {
            #[cfg(feature = "usrp")]
            SdrHandle::Usrp(h) => h.overflow_count(),
            #[cfg(feature = "hackrf")]
            SdrHandle::HackRf(h) => h.overflow_count(),
            #[cfg(feature = "bladerf")]
            SdrHandle::BladeRf(h) => h.overflow_count(),
            #[cfg(feature = "soapysdr")]
            SdrHandle::Soapy(h) => h.overflow_count(),
            #[cfg(feature = "aaronia")]
            SdrHandle::Aaronia(h) => h.overflow_count(),
            #[cfg(feature = "rfnm")]
            SdrHandle::Rfnm(h) => h.overflow_count(),
            #[cfg(feature = "sidekiq")]
            SdrHandle::Sidekiq(h) => h.overflow_count(),
            SdrHandle::Vita49(h) => h.overflow_count(),
        }
    }

    /// Aaronia-specific: cumulative counts of all warn-flag bits the device
    /// has set (overflow, dropped, inaccurate, resampled, time_disc). Returns
    /// None for other backends.
    fn aaronia_flag_counts(&self) -> Option<(u64, u64, u64, u64, u64)> {
        match self {
            #[cfg(feature = "aaronia")]
            SdrHandle::Aaronia(h) => Some(h.flag_counts()),
            _ => None,
        }
    }

    /// Actual sample rate in Hz. Most SDRs match the requested rate exactly.
    /// RFNM may differ (e.g., 122.88 Msps when 122 Msps was requested).
    /// Aaronia: read from packet.stepFrequency at open time per vendor docs.
    fn actual_sample_rate(&self, requested: u32) -> u64 {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "rfnm")]
            SdrHandle::Rfnm(h) => h.actual_sample_rate(),
            SdrHandle::Vita49(h) => h.sample_rate() as u64,
            #[cfg(feature = "aaronia")]
            SdrHandle::Aaronia(h) => h.actual_sample_rate() as u64,
            _ => requested as u64,
        }
    }

    /// Set SDR gain at runtime. For HackRF, gain is split as LNA=gain, VGA from lna/vga fields.
    #[allow(unused_variables)]
    fn set_gain(&self, gain: f64, hackrf_lna: Option<u32>, hackrf_vga: Option<u32>) {
        match self {
            #[cfg(feature = "usrp")]
            SdrHandle::Usrp(h) => h.set_gain(gain),
            #[cfg(feature = "hackrf")]
            SdrHandle::HackRf(h) => {
                let lna = hackrf_lna.unwrap_or(gain as u32);
                let vga = hackrf_vga.unwrap_or(20);
                h.set_gain(lna, vga);
            }
            #[cfg(feature = "bladerf")]
            SdrHandle::BladeRf(h) => h.set_gain(gain),
            #[cfg(feature = "soapysdr")]
            SdrHandle::Soapy(h) => h.set_gain(gain),
            #[cfg(feature = "aaronia")]
            SdrHandle::Aaronia(h) => h.set_gain(gain),
            #[cfg(feature = "rfnm")]
            SdrHandle::Rfnm(h) => h.set_gain(gain),
            #[cfg(feature = "sidekiq")]
            SdrHandle::Sidekiq(h) => h.set_gain(gain),
            SdrHandle::Vita49(h) => h.set_gain(gain),
        }
    }
}

/// Open the appropriate SDR handle based on interface string.
#[allow(unused_variables)]
fn open_sdr_handle(
    iface: &str,
    sample_rate: u32,
    center_freq_hz: u64,
    gain: f64,
    hackrf_lna: u32,
    hackrf_vga: u32,
    antenna: Option<&str>,
    aaronia_decim: u32,
    sidekiq_agc: bool,
    sidekiq_dc_corr: bool,
) -> Result<SdrHandle, String> {
    let sdr_type = detect_sdr_type(iface);
    match sdr_type {
        SdrType::Vita49 => {
            let h = bd_sdr::vita49::Vita49Handle::open(iface, sample_rate, center_freq_hz, gain)?;
            Ok(SdrHandle::Vita49(h))
        }
        #[cfg(feature = "usrp")]
        SdrType::Usrp => {
            let h = bd_sdr::usrp::UsrpHandle::open(iface, sample_rate, center_freq_hz, gain, antenna)?;
            Ok(SdrHandle::Usrp(h))
        }
        #[cfg(feature = "hackrf")]
        SdrType::HackRf => {
            let h = bd_sdr::hackrf::HackrfHandle::open(
                iface, sample_rate, center_freq_hz, hackrf_lna, hackrf_vga,
            )?;
            Ok(SdrHandle::HackRf(h))
        }
        #[cfg(feature = "bladerf")]
        SdrType::BladeRf => {
            let h = bd_sdr::bladerf::BladerfHandle::open(
                iface, sample_rate, center_freq_hz, gain as i32, antenna,
            )?;
            Ok(SdrHandle::BladeRf(h))
        }
        #[cfg(feature = "soapysdr")]
        SdrType::SoapySdr => {
            let h = bd_sdr::soapysdr::SoapyHandle::open(
                iface, sample_rate, center_freq_hz, gain,
            )?;
            Ok(SdrHandle::Soapy(h))
        }
        #[cfg(feature = "aaronia")]
        SdrType::Aaronia => {
            let h = bd_sdr::aaronia::AaroniaHandle::open(
                iface, sample_rate, center_freq_hz, gain, antenna, aaronia_decim,
            )?;
            Ok(SdrHandle::Aaronia(h))
        }
        #[cfg(feature = "rfnm")]
        SdrType::Rfnm => {
            let h = bd_sdr::rfnm::RfnmHandle::open(
                iface, sample_rate, center_freq_hz, gain, antenna,
            )?;
            Ok(SdrHandle::Rfnm(h))
        }
        #[cfg(feature = "sidekiq")]
        SdrType::Sidekiq => {
            let extras = bd_sdr::sidekiq::SidekiqExtras {
                agc: sidekiq_agc,
                dc_offset_corr: sidekiq_dc_corr,
                boost_priority: true,
            };
            let h = bd_sdr::sidekiq::SidekiqHandle::open_with_extras(
                iface, sample_rate, center_freq_hz, gain, antenna, extras,
            )?;
            Ok(SdrHandle::Sidekiq(h))
        }
        #[allow(unreachable_patterns)]
        _ => Err(format!(
            "unsupported SDR type '{}' (interface: '{}'). Compile with the appropriate feature flag.",
            sdr_type, iface,
        )),
    }
}

/// Configuration for live SDR capture.
pub struct LiveConfig<'a> {
    pub iface: &'a str,
    pub center_freq_mhz: u32,
    pub num_channels: usize,
    pub gain: f64,
    pub squelch_db: f32,
    pub hackrf_lna: u32,
    pub hackrf_vga: u32,
    /// Sidekiq: use AD9361 AGC instead of the manual gain-index path.
    pub sidekiq_agc: bool,
    /// Sidekiq: enable FPGA DC offset correction.
    pub sidekiq_dc_corr: bool,
    pub antenna: Option<&'a str>,
    pub pcap_path: Option<&'a Path>,
    pub burst_path: Option<&'a Path>,
    pub burst_limit_bytes: u64,
    pub check_crc: bool,
    pub print_stats: bool,
    pub use_gpu: bool,
    pub zmq_endpoint: Option<&'a str>,
    pub zmq_curve_keyfile: Option<&'a str>,
    pub sensor_id: Option<&'a str>,
    pub gpsd_enabled: bool,
    pub hci_enabled: bool,
    pub active_scan: bool,
    pub coded_scan: bool,
    pub classic_uaps: &'a [(u32, u8)],
    /// Aaronia-only: decimation factor (1=Full, 2=halfband DC notch, ...).
    /// Ignored by other backends.
    pub aaronia_decim: u32,
    pub running: Arc<AtomicBool>,
}

/// Run live SDR capture pipeline.
pub fn run_live(cfg: LiveConfig<'_>) -> Result<(), String> {
    let LiveConfig {
        iface,
        center_freq_mhz,
        num_channels,
        gain,
        squelch_db,
        hackrf_lna,
        hackrf_vga,
        antenna,
        pcap_path,
        burst_path,
        burst_limit_bytes,
        check_crc,
        print_stats,
        use_gpu,
        zmq_endpoint,
        zmq_curve_keyfile,
        sensor_id,
        gpsd_enabled,
        hci_enabled,
        active_scan,
        coded_scan,
        classic_uaps,
        aaronia_decim,
        sidekiq_agc,
        sidekiq_dc_corr,
        running,
    } = cfg;
    let sample_rate = num_channels as u32 * 1_000_000;
    let center_freq_hz = center_freq_mhz as u64 * 1_000_000;
    let edr_wideband_enabled = std::env::var_os("BD_EDR_WIDEBAND").is_some();

    // Per-channel adaptive gain is enabled for halfband (decim>1) mode
    // where it consistently improves CRC by ~7-8 points (attenuate-only
    // equalizes WiFi-loud bins down toward median). It is disabled for
    // Full (decim=1) mode where it is unstable across RF environments --
    // sometimes +8 points, sometimes -15. Stable baseline beats variance.
    let channel_gain_enabled = aaronia_decim > 1;
    let channel_gain_max = if channel_gain_enabled {
        CHANNEL_GAIN_MAX_DECIM
    } else {
        1.0 // unused when disabled, but keep type happy
    };

    let (channel_freqs, live_ch, first_live, last_live) =
        build_channel_map(center_freq_mhz, num_channels)?;

    let active_channels = (first_live..=last_live)
        .filter(|&ch| live_ch[ch] >= 0)
        .count();
    let active_bt_channels = channel_freqs
        .iter()
        .filter(|&&freq| (2402..=2480).contains(&freq))
        .count();

    let sdr_type = detect_sdr_type(iface);

    eprintln!(
        "channels: {} FFT bins, {} Classic + {} BLE channels (ch {}-{}, {}-{} MHz), SDR: {}",
        num_channels,
        active_bt_channels,
        active_channels,
        first_live,
        last_live,
        2402 + first_live * 2,
        2402 + last_live * 2,
        sdr_type,
    );

    // Initialize protocol subsystems
    let aa_correlator = AaCorrelator::new(); // LE 1M: SPS=2
    let aa_correlator_2m = AaCorrelator::with_sps(1); // LE 2M: SPS=1
    let syndrome_map = SyndromeMap::new(1);
    let conn_table = ConnectionTable::new();

    let semi_len = 4;
    let prototype = window::pfb_prototype_float(num_channels, semi_len);
    let sps = 2usize;

    // Per-channel burst catchers
    // With --coded-scan, advertising channels get scan mode for continuous
    // coded PHY capture regardless of squelch.
    let burst_catchers: Vec<Option<BurstCatcher>> = channel_freqs
        .iter()
        .map(|&freq| {
            if (2402..=2480).contains(&freq) {
                let is_adv = matches!(freq, 2402 | 2426 | 2480);
                if coded_scan && is_adv {
                    Some(BurstCatcher::new_scan(freq, squelch_db))
                } else {
                    Some(BurstCatcher::new(freq, squelch_db))
                }
            } else {
                None
            }
        })
        .collect();

    // Open SDR early so we can query the actual sample rate for resample ratio.
    let mut sdr = open_sdr_handle(
        iface,
        sample_rate,
        center_freq_hz,
        gain,
        hackrf_lna,
        hackrf_vga,
        antenna,
        aaronia_decim,
        sidekiq_agc,
        sidekiq_dc_corr,
    )?;

    // Compute resample ratio: if actual per-channel rate differs from target
    // (sps * 1 MHz), resample demod output to correct timing drift.
    // RFNM: 122.88/61 = 2.0144 Msps, ratio = 2.0/2.0144 = 0.99283.
    // Other SDRs: actual == requested, ratio = 1.0 (no resampling).
    let actual_rate = sdr.actual_sample_rate(sample_rate);
    let actual_channel_rate = actual_rate as f64 / (num_channels as f64 / 2.0);
    let target_channel_rate = sps as f64 * 1_000_000.0;
    let resample_ratio = target_channel_rate / actual_channel_rate;
    if (resample_ratio - 1.0).abs() > 0.001 {
        eprintln!(
            "FSK: resampling {:.4} → {:.4} Msps/ch (ratio={:.6})",
            actual_channel_rate / 1e6,
            target_channel_rate / 1e6,
            resample_ratio,
        );
    }

    let fsk = FskDemod::with_resample(sps, resample_ratio);

    let pcap_writer: Option<PcapWriter<BufWriter<File>>> = if let Some(path) = pcap_path {
        let file = File::create(path)
            .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
        let writer = BufWriter::new(file);
        Some(PcapWriter::new(writer).map_err(|e| format!("failed to write PCAP header: {}", e))?)
    } else {
        None
    };
    let burst_writer = open_burst_writer(burst_path, burst_limit_bytes)?;

    let fft_scale = 1.0 / num_channels as f32;

    // GPS client (optional)
    #[cfg(feature = "gps")]
    let gps_client: Option<bd_output::gps::GpsClient> = if gpsd_enabled {
        match bd_output::gps::GpsClient::new("localhost", 2947) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("GPS: {}", e);
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "gps"))]
    let _gps_client: Option<()> = {
        let _ = gpsd_enabled;
        None
    };

    // HCI GATT prober (optional, requires --hci flag)
    #[cfg(feature = "hci")]
    let hci_prober: Option<bd_hci::HciProber> = if hci_enabled {
        match bd_hci::HciProber::new() {
            Ok(prober) => {
                if prober.is_available() {
                    eprintln!("HCI: adapter available, GATT probing enabled");
                    Some(prober)
                } else {
                    eprintln!("HCI: no powered Bluetooth adapter found");
                    None
                }
            }
            Err(e) => {
                eprintln!("HCI: {}", e);
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "hci"))]
    let _ = hci_enabled;

    // HCI active scanner (optional, requires --active-scan flag)
    #[cfg(feature = "hci")]
    let scan_rx: Option<crossbeam::channel::Receiver<bd_hci::ScanResult>> = if active_scan {
        match bd_hci::HciScanner::new() {
            Ok(scanner) => {
                let (tx, rx) = crossbeam::channel::bounded(256);
                let scan_running = running.clone();
                std::thread::Builder::new()
                    .name("hci-scan".to_string())
                    .spawn(move || scanner.run(tx, scan_running))
                    .map_err(|e| format!("hci-scan thread: {}", e))?;
                eprintln!("HCI scan: active scanning thread started");
                Some(rx)
            }
            Err(e) => {
                eprintln!("HCI scan: {}", e);
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "hci"))]
    let _ = active_scan;
    #[cfg(all(feature = "hci", not(feature = "zmq")))]
    let _ = scan_rx;

    // ZMQ config to pass to processing thread (created there since zmq::Socket is !Send)
    #[cfg(feature = "zmq")]
    let zmq_config: Option<(String, Option<String>, Option<String>)> = zmq_endpoint.map(|ep| {
        (
            ep.to_string(),
            sensor_id.map(|s| s.to_string()),
            zmq_curve_keyfile.map(|s| s.to_string()),
        )
    });
    #[cfg(not(feature = "zmq"))]
    {
        let _ = (zmq_endpoint, zmq_curve_keyfile, sensor_id);
    }

    // Shared atomics for runtime C2 control
    let gain_pending = Arc::new(AtomicI32::new(i32::MIN));
    let squelch_pending = Arc::new(AtomicI32::new(i32::MIN));

    // C2 control thread + command dispatch (optional, requires ZMQ)
    #[cfg(feature = "zmq")]
    let hb_state_for_decode: Option<Arc<Mutex<bd_output::control::HeartbeatState>>>;
    #[cfg(feature = "zmq")]
    let _c2_threads: Option<(std::thread::JoinHandle<()>, std::thread::JoinHandle<()>)> =
        if let Some(ref ep) = zmq_endpoint {
            let ctrl_ep = bd_output::zmq_pub::derive_control_endpoint(ep);
            let sid = sensor_id.unwrap_or("blue-dragon").to_string();

            let hb_state = Arc::new(Mutex::new(bd_output::control::HeartbeatState::new(
                &sid,
                &sdr_type.to_string(),
                center_freq_mhz,
                num_channels as u32,
            )));
            {
                let mut s = hb_state.lock().unwrap_or_else(|e| e.into_inner());
                s.gain = gain;
                if sdr_type == SdrType::HackRf {
                    s.hackrf_lna = Some(hackrf_lna);
                    s.hackrf_vga = Some(hackrf_vga);
                }
                s.squelch = squelch_db;
            }

            let (cmd_tx, cmd_rx) = crossbeam::channel::bounded(16);
            let c2_running = running.clone();
            let curve_kf = zmq_curve_keyfile.map(|s| s.to_string());

            let ctrl = bd_output::control::ControlClient::new(
                &ctrl_ep,
                &sid,
                curve_kf.as_deref(),
                hb_state.clone(),
                cmd_tx,
                c2_running,
            );

            match ctrl {
                Ok(client) => {
                    hb_state_for_decode = Some(hb_state.clone());
                    let c2_handle = std::thread::Builder::new()
                        .name("c2-control".to_string())
                        .spawn(move || client.run())
                        .map_err(|e| format!("c2 thread: {}", e))?;

                    // Command dispatch thread: reads commands from C2, updates atomics
                    let gp = gain_pending.clone();
                    let sp = squelch_pending.clone();
                    let hb = hb_state.clone();
                    let disp_running = running.clone();

                    // Move HCI prober into dispatch thread if available
                    #[cfg(feature = "hci")]
                    let dispatch_hci = hci_prober;
                    // ZMQ publisher for GATT results (created in dispatch thread)
                    #[cfg(feature = "hci")]
                    let dispatch_zmq_ep = zmq_endpoint.map(|s| s.to_string());
                    #[cfg(feature = "hci")]
                    let dispatch_zmq_sid = sensor_id.map(|s| s.to_string());
                    #[cfg(feature = "hci")]
                    let dispatch_zmq_curve = zmq_curve_keyfile.map(|s| s.to_string());

                    let dispatch_handle = std::thread::Builder::new()
                        .name("c2-dispatch".to_string())
                        .spawn(move || {
                            use bd_output::control::ControlCommand;

                            // Create a separate ZMQ publisher for GATT results
                            #[cfg(feature = "hci")]
                            let gatt_zmq_pub: Option<bd_output::zmq_pub::ZmqPublisher> =
                                dispatch_zmq_ep.and_then(|ep| {
                                    bd_output::zmq_pub::ZmqPublisher::new(
                                        &ep,
                                        dispatch_zmq_sid.as_deref(),
                                        dispatch_zmq_curve.as_deref(),
                                    ).ok()
                                });

                            while disp_running.load(Ordering::Relaxed) {
                                match cmd_rx.recv_timeout(std::time::Duration::from_secs(1)) {
                                    Ok(cmd) => match cmd {
                                        ControlCommand::SetGain { gain, lna, vga, req_id: _ } => {
                                            // Store gain * 10 as integer
                                            gp.store((gain * 10.0) as i32, Ordering::Relaxed);
                                            if let Ok(mut s) = hb.lock() {
                                                s.gain = gain;
                                                if let Some(l) = lna {
                                                    s.hackrf_lna = Some(l as u32);
                                                }
                                                if let Some(v) = vga {
                                                    s.hackrf_vga = Some(v as u32);
                                                }
                                            }
                                            eprintln!("C2: gain set to {:.1} dB", gain);
                                        }
                                        ControlCommand::SetSquelch { threshold, req_id: _ } => {
                                            sp.store((threshold * 10.0) as i32, Ordering::Relaxed);
                                            if let Ok(mut s) = hb.lock() {
                                                s.squelch = threshold;
                                            }
                                            eprintln!("C2: squelch set to {:.1} dB", threshold);
                                        }
                                        ControlCommand::Restart { center_freq: _, channels: _, req_id: _ } => {
                                            eprintln!("C2: restart requested (not yet implemented in Rust)");
                                        }
                                        #[cfg(feature = "hci")]
                                        ControlCommand::QueryGatt { mac, req_id: _ } => {
                                            if let Some(ref prober) = dispatch_hci {
                                                eprintln!("C2: GATT query for {}", mac);
                                                let result = prober.query(&mac);
                                                if let Some(ref e) = result.error {
                                                    eprintln!("C2: GATT error for {}: {}", mac, e);
                                                } else {
                                                    eprintln!("C2: GATT {} services for {}",
                                                        result.services.len(), mac);
                                                }
                                                if let Some(ref pub_socket) = gatt_zmq_pub {
                                                    if let Ok(val) = serde_json::to_value(&result) {
                                                        pub_socket.send_gatt(&val);
                                                    }
                                                }
                                            } else {
                                                eprintln!("C2: GATT query for {} ignored (no HCI adapter)", mac);
                                            }
                                        }
                                        #[cfg(not(feature = "hci"))]
                                        ControlCommand::QueryGatt { .. } => {
                                            eprintln!("C2: GATT query ignored (compiled without hci feature)");
                                        }
                                        #[cfg(feature = "hci")]
                                        ControlCommand::WriteGatt { mac, char_uuid, service_uuid, data, req_id: _ } => {
                                            if let Some(ref prober) = dispatch_hci {
                                                eprintln!("C2: GATT write {} bytes to {} on {}",
                                                    data.len(), char_uuid, mac);
                                                let result = prober.write_characteristic(
                                                    &mac,
                                                    service_uuid.as_deref(),
                                                    &char_uuid,
                                                    &data,
                                                );
                                                if result.success {
                                                    eprintln!("C2: GATT write success on {} (svc {})",
                                                        mac, result.service_uuid);
                                                } else {
                                                    eprintln!("C2: GATT write failed on {}: {}",
                                                        mac, result.error.as_deref().unwrap_or("unknown"));
                                                }
                                                if let Some(ref pub_socket) = gatt_zmq_pub {
                                                    if let Ok(val) = serde_json::to_value(&result) {
                                                        pub_socket.send_gatt(&val);
                                                    }
                                                }
                                            } else {
                                                eprintln!("C2: GATT write for {} ignored (no HCI adapter)", mac);
                                            }
                                        }
                                        #[cfg(not(feature = "hci"))]
                                        ControlCommand::WriteGatt { .. } => {
                                            eprintln!("C2: GATT write ignored (compiled without hci feature)");
                                        }
                                        #[cfg(feature = "hci")]
                                        ControlCommand::SpoofAdv { name, service_uuids, connectable, duration_secs, req_id: _ } => {
                                            eprintln!("C2: spoof advertisement (name={:?}, connectable={}, {}s)",
                                                name, connectable, duration_secs);
                                            let config = bd_hci::SpoofAdvConfig {
                                                name,
                                                manufacturer_data: std::collections::HashMap::new(),
                                                service_uuids,
                                                service_data: std::collections::HashMap::new(),
                                                tx_power: None,
                                                connectable,
                                                duration_secs,
                                            };
                                            // Run in a separate thread to not block C2 dispatch
                                            std::thread::Builder::new()
                                                .name("spoof-adv".to_string())
                                                .spawn(move || {
                                                    let result = bd_hci::spoof_advertisement(&config);
                                                    if !result.success {
                                                        eprintln!("C2: spoof failed: {}",
                                                            result.error.unwrap_or_default());
                                                    }
                                                })
                                                .ok();
                                        }
                                        #[cfg(not(feature = "hci"))]
                                        ControlCommand::SpoofAdv { .. } => {
                                            eprintln!("C2: spoof ignored (compiled without hci feature)");
                                        }
                                        #[cfg(feature = "hci")]
                                        ControlCommand::L2capFlood { mac, count, hold_secs, psm, req_id: _ } => {
                                            eprintln!("C2: L2CAP flood {} x{} hold={}s psm={}",
                                                mac, count, hold_secs, psm);
                                            let config = bd_hci::L2capFloodConfig {
                                                mac,
                                                count,
                                                hold_secs,
                                                psm,
                                            };
                                            std::thread::Builder::new()
                                                .name("l2cap-flood".to_string())
                                                .spawn(move || {
                                                    let result = bd_hci::l2cap_flood(&config);
                                                    eprintln!("C2: L2CAP flood done: {} open, {} failed",
                                                        result.connections_opened, result.connections_failed);
                                                })
                                                .ok();
                                        }
                                        #[cfg(not(feature = "hci"))]
                                        ControlCommand::L2capFlood { .. } => {
                                            eprintln!("C2: L2CAP flood ignored (compiled without hci feature)");
                                        }
                                    },
                                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                                    Err(_) => break,
                                }
                            }
                        })
                        .map_err(|e| format!("c2 dispatch thread: {}", e))?;

                    Some((c2_handle, dispatch_handle))
                }
                Err(e) => {
                    hb_state_for_decode = None;
                    eprintln!("C2: {}", e);
                    None
                }
            }
        } else {
            hb_state_for_decode = None;
            None
        };

    // HCI active-scan forwarder thread: drains scan_rx, publishes on "scan:" topic
    #[cfg(all(feature = "hci", feature = "zmq"))]
    let _scan_forwarder: Option<std::thread::JoinHandle<()>> = scan_rx.and_then(|rx| {
        let ep = zmq_endpoint?.to_string();
        let sid = sensor_id.map(|s| s.to_string());
        let curve_kf = zmq_curve_keyfile.map(|s| s.to_string());
        let scan_running = running.clone();
        Some(
            std::thread::Builder::new()
                .name("scan-fwd".to_string())
                .spawn(move || {
                    let pub_socket = match bd_output::zmq_pub::ZmqPublisher::new(
                        &ep,
                        sid.as_deref(),
                        curve_kf.as_deref(),
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("HCI scan ZMQ: {}", e);
                            return;
                        }
                    };
                    while scan_running.load(std::sync::atomic::Ordering::Relaxed) {
                        match rx.recv_timeout(std::time::Duration::from_secs(1)) {
                            Ok(result) => {
                                if let Ok(val) = serde_json::to_value(&result) {
                                    pub_socket.send_scan(&val);
                                }
                            }
                            Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                            Err(_) => break,
                        }
                    }
                })
                .expect("failed to spawn scan-fwd thread"),
        )
    });

    // GPU path
    #[cfg(feature = "gpu")]
    if use_gpu {
        if edr_wideband_enabled {
            return Err("BD_EDR_WIDEBAND currently requires --no-gpu".to_string());
        }
        return run_live_gpu_loop(
            sdr,
            &running,
            num_channels,
            semi_len,
            &prototype,
            fft_scale,
            channel_gain_max,
            channel_gain_enabled,
            burst_catchers,
            fsk,
            aa_correlator,
            aa_correlator_2m,
            syndrome_map,
            conn_table,
            classic_uaps.to_vec(),
            pcap_writer,
            burst_writer,
            check_crc,
            print_stats,
            gain_pending.clone(),
            squelch_pending.clone(),
            #[cfg(feature = "zmq")]
            zmq_config,
            #[cfg(feature = "zmq")]
            hb_state_for_decode,
            #[cfg(feature = "gps")]
            gps_client,
        );
    }

    #[cfg(not(feature = "gpu"))]
    let _ = use_gpu;

    // CPU path architecture (matches C tool):
    //   SDR recv thread -> [bounded channel] -> PFB+FFT thread
    //   -> [broadcast] -> parallel burst workers -> decode thread
    use std::sync::atomic::AtomicU64;

    let max_samps = sdr.max_samps();

    let overflow_count = Arc::new(AtomicU64::new(0));

    eprintln!("CPU: PFB+FFT (SIMD), max_recv={}", max_samps);

    // Spawn parallel burst workers + decode thread
    let (batch_txs, worker_handles, decode_handle) = spawn_parallel_pipeline(
        num_channels,
        center_freq_mhz,
        sample_rate,
        edr_wideband_enabled,
        1.0, // CPU path pre-scales by fft_scale, workers use 1.0
        channel_gain_max,
        channel_gain_enabled,
        burst_catchers,
        fsk,
        aa_correlator,
        aa_correlator_2m,
        syndrome_map,
        conn_table,
        classic_uaps.to_vec(),
        pcap_writer,
        burst_writer,
        check_crc,
        print_stats,
        overflow_count.clone(),
        squelch_pending.clone(),
        #[cfg(feature = "zmq")]
        zmq_config,
        #[cfg(feature = "zmq")]
        hb_state_for_decode,
        #[cfg(feature = "gps")]
        gps_client,
    );

    // SDR recv thread: continuously drains hardware, sends i16 buffers to PFB thread.
    // Accumulates multiple recv calls to ~4080 complex samples before sending,
    // compensating for smaller max_samps with SC16 wire format.
    let (sdr_tx, sdr_rx) = channel::bounded::<Vec<i16>>(32);
    let sdr_overflow = overflow_count.clone();
    let sdr_running = running.clone();
    let sdr_buf_size = max_samps * 2; // i16 elements per recv

    let sdr_gain_pending = gain_pending.clone();
    let sdr_thread = std::thread::Builder::new()
        .name("sdr-recv".to_string())
        .spawn(move || {
            let target_samples = max_samps.max(4080);
            let target_elems = target_samples * 2;
            let mut recv_buf = vec![0i16; sdr_buf_size];
            let mut send_buf = vec![0i16; target_elems + sdr_buf_size];
            let mut send_pos: usize = 0;
            let mut last_flag_log = Instant::now();
            let mut prev_flags: (u64, u64, u64, u64, u64) = (0, 0, 0, 0, 0);

            while sdr_running.load(Ordering::Relaxed) {
                // Check for runtime gain change from C2
                let pending_gain = sdr_gain_pending.swap(i32::MIN, Ordering::Relaxed);
                if pending_gain != i32::MIN {
                    let gain_db = pending_gain as f64 / 10.0;
                    sdr.set_gain(gain_db, None, None);
                }

                let num_rx = sdr.recv_into_i16(&mut recv_buf);
                if num_rx == 0 {
                    continue;
                }
                sdr_overflow.store(sdr.overflow_count(), Ordering::Relaxed);

                if last_flag_log.elapsed().as_secs() >= 5 {
                    if let Some(cur) = sdr.aaronia_flag_counts() {
                        let d = (
                            cur.0 - prev_flags.0,
                            cur.1 - prev_flags.1,
                            cur.2 - prev_flags.2,
                            cur.3 - prev_flags.3,
                            cur.4 - prev_flags.4,
                        );
                        if d.0 + d.1 + d.2 + d.3 + d.4 > 0 {
                            log::warn!(
                                "Aaronia warn flags (5s deltas): overflow={} dropped={} inaccurate={} resampled={} time_disc={}",
                                d.0, d.1, d.2, d.3, d.4,
                            );
                        }
                        prev_flags = cur;
                    }
                    last_flag_log = Instant::now();
                }
                let n = num_rx * 2;
                send_buf[send_pos..send_pos + n].copy_from_slice(&recv_buf[..n]);
                send_pos += n;

                if send_pos >= target_elems {
                    if sdr_tx.send(send_buf[..send_pos].to_vec()).is_err() {
                        break;
                    }
                    send_pos = 0;
                }
            }
        })
        .expect("failed to spawn sdr-recv thread");

    // PFB+FFT processing thread (main thread)
    let mut channelizer = PfbChannelizer::new(num_channels, semi_len, &prototype);
    let mut fft = BatchFft::new(num_channels);
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); num_channels];
    let mut float_tmp = vec![0.0f32; num_channels * 2];

    const CPU_BATCH_STEPS: usize = 4096;
    let batch_floats = CPU_BATCH_STEPS * num_channels * 2;
    let mut batch = Vec::with_capacity(batch_floats);
    let mut raw_batch = edr_wideband_enabled
        .then(|| Vec::with_capacity(CPU_BATCH_STEPS * num_channels));
    let mut batch_steps: usize = 0;
    let mut next_batch_ts: Option<Timespec> = None;

    for i16_buf in sdr_rx.iter() {
        let n = i16_buf.len();

        let step = num_channels;
        let num_blocks = n / step;

        for block in 0..num_blocks {
            let offset = block * step;

            channelizer.execute_into(&i16_buf[offset..offset + step], &mut fft_buf);
            fft.process(&mut fft_buf);

            for (j, val) in fft_buf.iter().enumerate() {
                float_tmp[j * 2] = val.re * fft_scale;
                float_tmp[j * 2 + 1] = val.im * fft_scale;
            }
            batch.extend_from_slice(&float_tmp);
            if let Some(raw) = raw_batch.as_mut() {
                raw.extend_from_slice(&i16_buf[offset..offset + step]);
            }
            batch_steps += 1;

            if batch.len() >= batch_floats {
                let ts = next_batch_ts
                    .unwrap_or_else(|| initial_batch_timestamp(batch_steps));
                next_batch_ts = Some(channel_samples_after(&ts, batch_steps));
                broadcast_batch(
                    &batch_txs,
                    std::mem::replace(&mut batch, Vec::with_capacity(batch_floats)),
                    raw_batch
                        .as_mut()
                        .map(|raw| std::mem::replace(raw, Vec::with_capacity(CPU_BATCH_STEPS * num_channels))),
                    batch_steps,
                    &ts,
                );
                batch_steps = 0;
            }
        }
    }

    if !batch.is_empty() {
        let ts = next_batch_ts.unwrap_or_else(|| initial_batch_timestamp(batch_steps));
        broadcast_batch(&batch_txs, batch, raw_batch, batch_steps, &ts);
    }

    drop(batch_txs);
    for h in worker_handles {
        let _ = h.join();
    }
    let _ = decode_handle.join();
    let _ = sdr_thread.join();

    Ok(())
}

/// GPU-accelerated live capture processing loop with recv threading.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn run_live_gpu_loop(
    sdr: SdrHandle,
    running: &Arc<AtomicBool>,
    num_channels: usize,
    semi_len: usize,
    prototype: &[f32],
    fft_scale: f32,
    channel_gain_max: f32,
    channel_gain_enabled: bool,
    burst_catchers: Vec<Option<BurstCatcher>>,
    fsk: FskDemod,
    aa_correlator: AaCorrelator,
    aa_correlator_2m: AaCorrelator,
    syndrome_map: SyndromeMap,
    conn_table: ConnectionTable,
    classic_uaps: Vec<(u32, u8)>,
    pcap_writer: Option<PcapWriter<BufWriter<File>>>,
    burst_writer: Option<FileBurstWriter>,
    check_crc: bool,
    print_stats: bool,
    gain_pending: Arc<AtomicI32>,
    squelch_pending: Arc<AtomicI32>,
    #[cfg(feature = "zmq")] zmq_config: Option<(String, Option<String>, Option<String>)>,
    #[cfg(feature = "zmq")] hb_state_for_decode: Option<
        Arc<Mutex<bd_output::control::HeartbeatState>>,
    >,
    #[cfg(feature = "gps")] gps_client: Option<bd_output::gps::GpsClient>,
) -> Result<(), String> {
    use std::sync::atomic::AtomicU64;

    const GPU_BATCH_SIZE: usize = 4096;

    let sdr = sdr;
    let max_samps = sdr.max_samps();

    let mut gpu = bd_gpu::GpuChannelizer::new(num_channels, semi_len, prototype, GPU_BATCH_SIZE)?;

    let buffer_len = gpu.buffer_len();
    eprintln!(
        "GPU: batch={} buffer={}KB result={}KB max_recv={}",
        GPU_BATCH_SIZE,
        buffer_len / 1024,
        (GPU_BATCH_SIZE * num_channels * 8) / 1024,
        max_samps
    );

    let overflow_count = Arc::new(AtomicU64::new(0));

    // Spawn parallel burst workers + decode thread
    let (batch_txs, worker_handles, decode_handle) = spawn_parallel_pipeline(
        num_channels,
        0,
        num_channels as u32 * 1_000_000,
        false,
        fft_scale, // GPU output is raw, workers apply fft_scale
        channel_gain_max,
        channel_gain_enabled,
        burst_catchers,
        fsk,
        aa_correlator,
        aa_correlator_2m,
        syndrome_map,
        conn_table,
        classic_uaps,
        pcap_writer,
        burst_writer,
        check_crc,
        print_stats,
        overflow_count.clone(),
        squelch_pending,
        #[cfg(feature = "zmq")]
        zmq_config,
        #[cfg(feature = "zmq")]
        hb_state_for_decode,
        #[cfg(feature = "gps")]
        gps_client,
    );

    // SDR recv thread: continuously drains hardware, prevents overflow during GPU submit.
    // Accumulates multiple recv calls to ~4080 complex samples before sending.
    let (sdr_tx, sdr_rx) = channel::bounded::<Vec<i16>>(32);
    let sdr_overflow = overflow_count.clone();
    let sdr_running = running.clone();
    let sdr_buf_size = max_samps * 2;

    let sdr_gain_pending = gain_pending;
    let sdr_thread = std::thread::Builder::new()
        .name("sdr-recv-gpu".to_string())
        .spawn(move || {
            let mut sdr = sdr;
            let target_samples = max_samps.max(4080);
            let target_elems = target_samples * 2;
            let mut recv_buf = vec![0i16; sdr_buf_size];
            let mut send_buf = vec![0i16; target_elems + sdr_buf_size];
            let mut send_pos: usize = 0;

            while sdr_running.load(Ordering::Relaxed) {
                // Check for runtime gain change from C2
                let pending_gain = sdr_gain_pending.swap(i32::MIN, Ordering::Relaxed);
                if pending_gain != i32::MIN {
                    let gain_db = pending_gain as f64 / 10.0;
                    sdr.set_gain(gain_db, None, None);
                }

                let num_rx = sdr.recv_into_i16(&mut recv_buf);
                if num_rx == 0 {
                    continue;
                }
                sdr_overflow.store(sdr.overflow_count(), Ordering::Relaxed);
                let n = num_rx * 2;
                send_buf[send_pos..send_pos + n].copy_from_slice(&recv_buf[..n]);
                send_pos += n;

                if send_pos >= target_elems {
                    if sdr_tx.send(send_buf[..send_pos].to_vec()).is_err() {
                        break;
                    }
                    send_pos = 0;
                }
            }
        })
        .expect("failed to spawn sdr-recv-gpu thread");

    let mut pos: usize = 0;
    let mut raw_buf = gpu.raw_buffer();
    let mut next_batch_ts: Option<Timespec> = None;

    for i16_buf in sdr_rx.iter() {
        // Copy i16 data into GPU raw buffer, handling partial fills
        let mut src_pos = 0usize;
        let n = i16_buf.len();
        while src_pos < n {
            let copy_len = (n - src_pos).min(buffer_len - pos);
            raw_buf[pos..pos + copy_len].copy_from_slice(&i16_buf[src_pos..src_pos + copy_len]);
            pos += copy_len;
            src_pos += copy_len;

            if pos >= buffer_len {
                if let Some(result) = gpu.submit() {
                    let ts = next_batch_ts
                        .unwrap_or_else(|| initial_batch_timestamp(GPU_BATCH_SIZE));
                    next_batch_ts = Some(channel_samples_after(&ts, GPU_BATCH_SIZE));
                    broadcast_batch(&batch_txs, result.to_vec(), None, GPU_BATCH_SIZE, &ts);
                }
                pos = 0;
                raw_buf = gpu.raw_buffer();
            }
        }
    }

    if pos > 0 {
        for i in pos..buffer_len {
            raw_buf[i] = 0;
        }
        if let Some(result) = gpu.submit() {
            let ts = next_batch_ts
                .unwrap_or_else(|| initial_batch_timestamp(GPU_BATCH_SIZE));
            next_batch_ts = Some(channel_samples_after(&ts, GPU_BATCH_SIZE));
            broadcast_batch(&batch_txs, result.to_vec(), None, GPU_BATCH_SIZE, &ts);
        }
    }

    if let Some(result) = gpu.flush() {
        let ts = next_batch_ts.unwrap_or_else(|| initial_batch_timestamp(GPU_BATCH_SIZE));
        broadcast_batch(&batch_txs, result.to_vec(), None, GPU_BATCH_SIZE, &ts);
    }

    drop(batch_txs);
    for h in worker_handles {
        let _ = h.join();
    }
    let _ = decode_handle.join();
    let _ = sdr_thread.join();

    Ok(())
}
