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
use bd_dsp::fsk::{self, FskDemod};
use bd_dsp::pfb::PfbChannelizer;
use bd_dsp::window;
use bd_output::pcap::PcapWriter;
use bd_protocol::ble::{self, AaCorrelator};
use bd_protocol::ble_connection::ConnectionTable;
use bd_protocol::btbb::{self, SyndromeMap};
use bd_protocol::Timespec;
use bd_sdr::file::{FileSource, SampleFormat};
use bd_sdr::SdrSource;

/// Run the full pipeline from IQ file to PCAP output.
pub fn run_file(
    file_path: &Path,
    format: SampleFormat,
    center_freq_mhz: u32,
    num_channels: usize,
    pcap_path: Option<&Path>,
    check_crc: bool,
    squelch_db: f32,
    print_stats: bool,
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

    // Map FFT bins to valid BLE/BT channels (even MHz, 2402-2480)
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

    eprintln!(
        "channels: {} FFT bins, {} BLE channels (ch {}-{}, {}-{} MHz)",
        num_channels,
        active_channels,
        first_live,
        last_live,
        2402 + first_live * 2,
        2402 + last_live * 2,
    );

    // Initialize protocol subsystems
    let aa_correlator = AaCorrelator::new();       // LE 1M: SPS=2
    let aa_correlator_2m = AaCorrelator::with_sps(1); // LE 2M: SPS=1
    let syndrome_map = SyndromeMap::new(1);
    let mut conn_table = ConnectionTable::new();

    // Initialize DSP -- Type 2 PFB matching C code (semi_len m=4)
    let semi_len = 4;
    let prototype = window::pfb_prototype_float(num_channels, semi_len);
    let mut channelizer = PfbChannelizer::new(num_channels, semi_len, &prototype);
    let mut fft = BatchFft::new(num_channels);
    let sps = 2usize; // 2 samples per symbol (type 2 PFB output rate)

    // Per-channel burst catchers (only for valid BLE channels)
    let mut burst_catchers: Vec<Option<BurstCatcher>> = (0..40)
        .map(|ch| {
            if ch >= first_live && ch <= last_live && live_ch[ch] >= 0 {
                Some(BurstCatcher::new(2402 + ch as u32 * 2, squelch_db))
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
        Some(
            PcapWriter::new(writer)
                .map_err(|e| format!("failed to write PCAP header: {}", e))?,
        )
    } else {
        None
    };

    // Stats
    let mut total_ble: u64 = 0;
    let mut total_bt: u64 = 0;
    let mut total_crc: u64 = 0;
    let mut valid_crc: u64 = 0;
    let mut total_bursts: u64 = 0;
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

    // Main processing loop
    for buf in rx.iter() {
        // Type 2 PFB: each call takes M int16 values (M/2 complex samples)
        let step = num_channels; // M int16 values per PFB call
        let num_blocks = buf.data.len() / step;

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

            // Feed each live BLE channel through its burst catcher
            for ch_idx in first_live..=last_live {
                let fft_bin = live_ch[ch_idx];
                if fft_bin < 0 {
                    continue;
                }
                let fft_bin = fft_bin as usize;
                let catcher = match burst_catchers[ch_idx].as_mut() {
                    Some(c) => c,
                    None => continue,
                };

                let sample = fft_buf[fft_bin];
                if let Some(burst) = catcher.execute(sample, &ts) {
                    total_bursts += 1;

                    // Skip very short bursts (< 132 samples, matching C)
                    if burst.samples.len() < 132 {
                        continue;
                    }

                    // FSK demodulate the burst
                    if let Some(fsk_result) = fsk.demodulate(&burst.samples) {
                        let freq = burst.freq;
                        let burst_ts = burst.timestamp.clone();
                        let rssi = burst.rssi_db as i32;
                        let noise = burst.noise_db as i32;

                        // Try Classic BT first
                        if let Some(bt_pkt) = btbb::detect(
                            &fsk_result.bits,
                            freq,
                            rssi,
                            noise,
                            burst_ts.clone(),
                            &syndrome_map,
                        ) {
                            total_bt += 1;
                            if let Some(ref mut writer) = pcap_writer {
                                let _ = writer.write_bt(&bt_pkt, None);
                            }
                        } else {
                            // Try BLE LE 1M preamble-first detection
                            let mut pkt = ble::ble_burst(
                                &fsk_result.bits,
                                freq,
                                burst_ts.clone(),
                                check_crc,
                                |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
                            );

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
                                let bits_2m = fsk::reslice(&fsk_result.demod, fsk_result.silence, 1);
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

                            // Try LE Coded PHY (long range)
                            if pkt.is_none() {
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

                                // Track CRC stats
                                if p.crc_checked {
                                    total_crc += 1;
                                    if p.crc_valid {
                                        valid_crc += 1;
                                    }
                                }

                                total_ble += 1;
                                if let Some(ref mut writer) = pcap_writer {
                                    let _ = writer.write_ble(&p, None);
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

/// Process a burst: FSK demod -> BLE/BT detect -> PCAP write + ZMQ publish
#[allow(clippy::too_many_arguments)]
fn process_burst(
    burst: &bd_dsp::burst::Burst,
    fsk: &mut FskDemod,
    aa_correlator: &AaCorrelator,
    aa_correlator_2m: &AaCorrelator,
    syndrome_map: &SyndromeMap,
    conn_table: &mut ConnectionTable,
    pcap_writer: &mut Option<PcapWriter<BufWriter<File>>>,
    #[cfg(feature = "zmq")] zmq_pub: &Option<bd_output::zmq_pub::ZmqPublisher>,
    gps_fix: Option<&bd_output::pcap::GpsFix>,
    check_crc: bool,
    stats: &mut PipelineStats,
) {
    stats.total_bursts += 1;

    if burst.samples.len() < 132 {
        return;
    }

    let fsk_result = match fsk.demodulate(&burst.samples) {
        Some(r) => r,
        None => return,
    };

    let freq = burst.freq;
    let burst_ts = burst.timestamp.clone();
    let rssi = burst.rssi_db as i32;
    let noise = burst.noise_db as i32;

    // Try Classic BT first
    if let Some(bt_pkt) = btbb::detect(
        &fsk_result.bits,
        freq,
        rssi,
        noise,
        burst_ts.clone(),
        syndrome_map,
    ) {
        stats.total_bt += 1;
        if let Some(ref mut writer) = pcap_writer {
            let _ = writer.write_bt(&bt_pkt, gps_fix);
        }
        #[cfg(feature = "zmq")]
        if let Some(ref pub_socket) = zmq_pub {
            pub_socket.send_bt(&bt_pkt, gps_fix);
        }
        return;
    }

    // Try BLE LE 1M preamble-first detection
    let mut pkt = ble::ble_burst(
        &fsk_result.bits,
        freq,
        burst_ts.clone(),
        check_crc,
        |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
    );

    // Fall back to LE 1M AA correlator
    if pkt.is_none() {
        pkt = aa_correlator.correlate(
            &fsk_result.demod,
            freq,
            burst_ts.clone(),
            check_crc,
        );
    }

    // Try LE 2M: reslice the demod at SPS=1 and try preamble-first
    if pkt.is_none() {
        let bits_2m = fsk::reslice(&fsk_result.demod, fsk_result.silence, 1);
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

    // Try LE Coded PHY (long range)
    if pkt.is_none() {
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
            let _ = writer.write_ble(&p, gps_fix);
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
        }
    }

    fn crc_pct(&self) -> f64 {
        if self.total_crc > 0 {
            (self.valid_crc as f64 / self.total_crc as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Message sent from main thread to burst worker threads.
struct BatchMsg {
    data: Arc<Vec<f32>>,
    batch_steps: usize,
    ts: Timespec,
}

/// Channel assignment for a single burst worker.
#[derive(Clone)]
struct ChannelAssignment {
    ch_idx: usize,
    fft_bin: usize,
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
    fft_scale: f32,
    live_ch: [i32; 40],
    first_live: usize,
    last_live: usize,
    mut burst_catchers: Vec<Option<BurstCatcher>>,
    fsk: FskDemod,
    aa_correlator: AaCorrelator,
    aa_correlator_2m: AaCorrelator,
    syndrome_map: SyndromeMap,
    conn_table: ConnectionTable,
    pcap_writer: Option<PcapWriter<BufWriter<File>>>,
    check_crc: bool,
    print_stats: bool,
    overflow_count: Arc<std::sync::atomic::AtomicU64>,
    squelch_pending: Arc<AtomicI32>,
    #[cfg(feature = "zmq")]
    zmq_config: Option<(String, Option<String>, Option<String>)>,
    #[cfg(feature = "zmq")]
    hb_state: Option<Arc<Mutex<bd_output::control::HeartbeatState>>>,
    #[cfg(feature = "gps")]
    gps_client: Option<bd_output::gps::GpsClient>,
) -> (
    Vec<channel::Sender<BatchMsg>>,
    Vec<std::thread::JoinHandle<()>>,
    std::thread::JoinHandle<()>,
) {
    use bd_dsp::burst::Burst;

    // Collect active channels
    let active: Vec<ChannelAssignment> = (first_live..=last_live)
        .filter_map(|ch_idx| {
            if live_ch[ch_idx] >= 0 {
                Some(ChannelAssignment {
                    ch_idx,
                    fft_bin: live_ch[ch_idx] as usize,
                })
            } else {
                None
            }
        })
        .collect();

    let n_workers = active.len().min(8).max(1);

    // Burst output channel: all workers send here, decode thread receives
    let (burst_tx, burst_rx) = channel::bounded::<Burst>(512);

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

        let burst_tx = burst_tx.clone();
        let num_ch = num_channels;
        let scale = fft_scale;
        let sq_pending = squelch_pending.clone();

        let handle = std::thread::Builder::new()
            .name(format!("burst-{}", worker_id))
            .spawn(move || {
                let mut current_squelch = i32::MIN;
                for msg in batch_rx.iter() {
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
                        for t in 0..msg.batch_steps {
                            let base = t * num_ch * 2;
                            let idx = base + assign.fft_bin * 2;
                            let sample = num_complex::Complex32::new(
                                msg.data[idx] * scale,
                                msg.data[idx + 1] * scale,
                            );
                            if let Some(burst) = catcher.execute(sample, &msg.ts) {
                                let _ = burst_tx.send(burst);
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

    eprintln!("pipeline: {} burst workers, {} channels", n_workers, active.len());

    // Decode thread: FSK demod + BLE/BT protocol decode + output
    let decode_handle = {
        let overflow_proc = overflow_count;
        let mut fsk = fsk;
        let aa_correlator = aa_correlator;
        let aa_correlator_2m = aa_correlator_2m;
        let syndrome_map = syndrome_map;
        let mut conn_table = conn_table;
        let mut pcap_writer = pcap_writer;

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

                for burst in burst_rx.iter() {
                    #[cfg(feature = "gps")]
                    let gps_fix = gps_client.as_ref().map(|c| c.get_fix());
                    #[cfg(not(feature = "gps"))]
                    let gps_fix: Option<bd_output::pcap::GpsFix> = None;
                    let gps_ref = gps_fix.as_ref().filter(|f| f.valid);

                    process_burst(
                        &burst,
                        &mut fsk,
                        &aa_correlator,
                        &aa_correlator_2m,
                        &syndrome_map,
                        &mut conn_table,
                        &mut pcap_writer,
                        #[cfg(feature = "zmq")]
                        &zmq_pub,
                        gps_ref,
                        check_crc,
                        &mut stats,
                    );

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
                            "[{:.1}s] BLE: {}{} BT: {} bursts: {} CRC: {:.1}% ({}/{}) conns: {} overflow: {}\n",
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
                        );

                        // Update heartbeat state for C2
                        #[cfg(feature = "zmq")]
                        if let Some(ref hb) = hb_state {
                            if let Ok(mut s) = hb.lock() {
                                s.total_pkts = stats.total_ble + stats.total_bt;
                                s.pkt_rate = (stats.total_ble + stats.total_bt) as f64 / elapsed;
                                s.crc_pct = stats.crc_pct();
                            }
                        }

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
                        "done ({:.1}s): BLE: {}{} BT: {} bursts: {} CRC: {:.1}% ({}/{}) overflow: {}",
                        elapsed,
                        stats.total_ble,
                        phy_str,
                        stats.total_bt,
                        stats.total_bursts,
                        stats.crc_pct(),
                        stats.valid_crc,
                        stats.total_crc,
                        overflows,
                    );
                }
            })
            .expect("failed to spawn decode thread")
    };

    (batch_txs, worker_handles, decode_handle)
}

/// Broadcast a batch to all worker threads.
#[inline]
fn broadcast_batch(
    txs: &[channel::Sender<BatchMsg>],
    data: Vec<f32>,
    batch_steps: usize,
    ts: &Timespec,
) {
    let arc = Arc::new(data);
    for tx in txs {
        let _ = tx.send(BatchMsg {
            data: arc.clone(),
            batch_steps,
            ts: ts.clone(),
        });
    }
}

/// Detect SDR backend type from interface string.
fn detect_sdr_type(iface: &str) -> &str {
    if iface.starts_with("usrp") {
        "usrp"
    } else if iface.starts_with("hackrf") {
        "hackrf"
    } else if iface.starts_with("bladerf") {
        "bladerf"
    } else if iface.starts_with("soapy") {
        "soapysdr"
    } else {
        "usrp" // default
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
}

// Safety: SDR C library handles are thread-safe (recv from one thread is fine).
// The raw pointer inside UsrpHandle/etc. is an opaque C handle that supports this.
unsafe impl Send for SdrHandle {}

impl SdrHandle {
    fn recv_into(&mut self, buf: &mut [i8]) -> usize {
        match self {
            #[cfg(feature = "usrp")]
            SdrHandle::Usrp(h) => h.recv_into(buf),
            #[cfg(feature = "hackrf")]
            SdrHandle::HackRf(h) => h.recv_into(buf),
            #[cfg(feature = "bladerf")]
            SdrHandle::BladeRf(h) => h.recv_into(buf),
            #[cfg(feature = "soapysdr")]
            SdrHandle::Soapy(h) => h.recv_into(buf),
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
) -> Result<SdrHandle, String> {
    let sdr_type = detect_sdr_type(iface);
    match sdr_type {
        #[cfg(feature = "usrp")]
        "usrp" => {
            let h = bd_sdr::usrp::UsrpHandle::open(iface, sample_rate, center_freq_hz, gain)?;
            Ok(SdrHandle::Usrp(h))
        }
        #[cfg(feature = "hackrf")]
        "hackrf" => {
            let h = bd_sdr::hackrf::HackrfHandle::open(
                iface, sample_rate, center_freq_hz, hackrf_lna, hackrf_vga,
            )?;
            Ok(SdrHandle::HackRf(h))
        }
        #[cfg(feature = "bladerf")]
        "bladerf" => {
            let h = bd_sdr::bladerf::BladerfHandle::open(
                iface, sample_rate, center_freq_hz, gain as i32,
            )?;
            Ok(SdrHandle::BladeRf(h))
        }
        #[cfg(feature = "soapysdr")]
        "soapysdr" => {
            let h = bd_sdr::soapysdr::SoapyHandle::open(
                iface, sample_rate, center_freq_hz, gain,
            )?;
            Ok(SdrHandle::Soapy(h))
        }
        _ => Err(format!(
            "unsupported SDR type '{}' (interface: '{}'). Compile with the appropriate feature flag.",
            sdr_type, iface,
        )),
    }
}

/// Run live SDR capture pipeline.
#[allow(clippy::too_many_arguments)]
pub fn run_live(
    iface: &str,
    center_freq_mhz: u32,
    num_channels: usize,
    gain: f64,
    squelch_db: f32,
    hackrf_lna: u32,
    hackrf_vga: u32,
    pcap_path: Option<&Path>,
    check_crc: bool,
    print_stats: bool,
    use_gpu: bool,
    zmq_endpoint: Option<&str>,
    zmq_curve_keyfile: Option<&str>,
    sensor_id: Option<&str>,
    gpsd_enabled: bool,
    hci_enabled: bool,
    running: Arc<AtomicBool>,
) -> Result<(), String> {
    let sample_rate = num_channels as u32 * 1_000_000;
    let center_freq_hz = center_freq_mhz as u64 * 1_000_000;

    let (_channel_freqs, live_ch, first_live, last_live) =
        build_channel_map(center_freq_mhz, num_channels)?;

    let active_channels = (first_live..=last_live)
        .filter(|&ch| live_ch[ch] >= 0)
        .count();

    let sdr_type = detect_sdr_type(iface);

    eprintln!(
        "channels: {} FFT bins, {} BLE channels (ch {}-{}, {}-{} MHz), SDR: {}",
        num_channels, active_channels, first_live, last_live,
        2402 + first_live * 2, 2402 + last_live * 2, sdr_type,
    );

    // Initialize protocol subsystems
    let aa_correlator = AaCorrelator::new();       // LE 1M: SPS=2
    let aa_correlator_2m = AaCorrelator::with_sps(1); // LE 2M: SPS=1
    let syndrome_map = SyndromeMap::new(1);
    let conn_table = ConnectionTable::new();

    let semi_len = 4;
    let prototype = window::pfb_prototype_float(num_channels, semi_len);
    let sps = 2usize;

    // Per-channel burst catchers
    let burst_catchers: Vec<Option<BurstCatcher>> = (0..40)
        .map(|ch| {
            if ch >= first_live && ch <= last_live && live_ch[ch] >= 0 {
                Some(BurstCatcher::new(2402 + ch as u32 * 2, squelch_db))
            } else {
                None
            }
        })
        .collect();

    let fsk = FskDemod::new(sps);

    let pcap_writer: Option<PcapWriter<BufWriter<File>>> = if let Some(path) = pcap_path {
        let file = File::create(path)
            .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
        let writer = BufWriter::new(file);
        Some(PcapWriter::new(writer)
            .map_err(|e| format!("failed to write PCAP header: {}", e))?)
    } else {
        None
    };

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

    // ZMQ config to pass to processing thread (created there since zmq::Socket is !Send)
    #[cfg(feature = "zmq")]
    let zmq_config: Option<(String, Option<String>, Option<String>)> =
        zmq_endpoint.map(|ep| {
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

            let hb_state = Arc::new(Mutex::new(
                bd_output::control::HeartbeatState::new(
                    &sid, sdr_type, center_freq_mhz, num_channels as u32,
                ),
            ));
            {
                let mut s = hb_state.lock().unwrap();
                s.gain = gain;
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
                                        ControlCommand::SetGain { gain, lna: _, vga: _, req_id: _ } => {
                                            // Store gain * 10 as integer
                                            gp.store((gain * 10.0) as i32, Ordering::Relaxed);
                                            if let Ok(mut s) = hb.lock() {
                                                s.gain = gain;
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

    // GPU path
    #[cfg(feature = "gpu")]
    if use_gpu {
        let sdr = open_sdr_handle(iface, sample_rate, center_freq_hz, gain, hackrf_lna, hackrf_vga)?;

        return run_live_gpu_loop(
            sdr, &running, num_channels, semi_len, &prototype, fft_scale,
            live_ch, first_live, last_live, burst_catchers,
            fsk, aa_correlator, aa_correlator_2m, syndrome_map, conn_table,
            pcap_writer, check_crc, print_stats,
            gain_pending.clone(), squelch_pending.clone(),
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

    let mut sdr = open_sdr_handle(iface, sample_rate, center_freq_hz, gain, hackrf_lna, hackrf_vga)?;
    let max_samps = sdr.max_samps();

    let overflow_count = Arc::new(AtomicU64::new(0));

    eprintln!("CPU: PFB+FFT (SIMD), max_recv={}", max_samps);

    // Spawn parallel burst workers + decode thread
    let (batch_txs, worker_handles, decode_handle) = spawn_parallel_pipeline(
        num_channels,
        1.0, // CPU path pre-scales by fft_scale, workers use 1.0
        live_ch,
        first_live,
        last_live,
        burst_catchers,
        fsk,
        aa_correlator,
        aa_correlator_2m,
        syndrome_map,
        conn_table,
        pcap_writer,
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

    // SDR recv thread: continuously drains hardware, sends SC8 buffers to PFB thread
    // 32 slots provides ~2.7ms buffering at 60 MHz, absorbing scheduling jitter
    let (sdr_tx, sdr_rx) = channel::bounded::<Vec<i8>>(32);
    let sdr_overflow = overflow_count.clone();
    let sdr_running = running.clone();
    let sdr_buf_size = max_samps * 2;

    let sdr_gain_pending = gain_pending.clone();
    let sdr_thread = std::thread::Builder::new()
        .name("sdr-recv".to_string())
        .spawn(move || {
            let mut buf = vec![0i8; sdr_buf_size];
            while sdr_running.load(Ordering::Relaxed) {
                // Check for runtime gain change from C2
                let pending_gain = sdr_gain_pending.swap(i32::MIN, Ordering::Relaxed);
                if pending_gain != i32::MIN {
                    let gain_db = pending_gain as f64 / 10.0;
                    sdr.set_gain(gain_db, None, None);
                }

                let num_rx = sdr.recv_into(&mut buf);
                if num_rx == 0 {
                    continue;
                }
                sdr_overflow.store(sdr.overflow_count(), Ordering::Relaxed);
                let n = num_rx * 2;
                let mut send_buf = vec![0i8; n];
                send_buf.copy_from_slice(&buf[..n]);
                if sdr_tx.send(send_buf).is_err() {
                    break;
                }
            }
        })
        .expect("failed to spawn sdr-recv thread");

    // PFB+FFT processing thread (main thread)
    let mut channelizer = PfbChannelizer::new(num_channels, semi_len, &prototype);
    let mut fft = BatchFft::new(num_channels);
    let mut i16_buf = vec![0i16; max_samps * 2];
    let mut fft_buf = vec![Complex32::new(0.0, 0.0); num_channels];
    let mut float_tmp = vec![0.0f32; num_channels * 2];

    const CPU_BATCH_STEPS: usize = 4096;
    let batch_floats = CPU_BATCH_STEPS * num_channels * 2;
    let mut batch = Vec::with_capacity(batch_floats);
    let mut batch_steps: usize = 0;

    for sc8_buf in sdr_rx.iter() {
        let n = sc8_buf.len();
        for i in 0..n {
            i16_buf[i] = (sc8_buf[i] as i16) << 8;
        }
        drop(sc8_buf); // free SC8 buffer immediately

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
            batch_steps += 1;

            if batch.len() >= batch_floats {
                let ts = {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default();
                    Timespec {
                        tv_sec: now.as_secs(),
                        tv_nsec: now.subsec_nanos() as u64,
                    }
                };
                broadcast_batch(
                    &batch_txs,
                    std::mem::replace(&mut batch, Vec::with_capacity(batch_floats)),
                    batch_steps,
                    &ts,
                );
                batch_steps = 0;
            }
        }
    }

    if !batch.is_empty() {
        let ts = {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default();
            Timespec {
                tv_sec: now.as_secs(),
                tv_nsec: now.subsec_nanos() as u64,
            }
        };
        broadcast_batch(&batch_txs, batch, batch_steps, &ts);
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
    live_ch: [i32; 40],
    first_live: usize,
    last_live: usize,
    burst_catchers: Vec<Option<BurstCatcher>>,
    fsk: FskDemod,
    aa_correlator: AaCorrelator,
    aa_correlator_2m: AaCorrelator,
    syndrome_map: SyndromeMap,
    conn_table: ConnectionTable,
    pcap_writer: Option<PcapWriter<BufWriter<File>>>,
    check_crc: bool,
    print_stats: bool,
    gain_pending: Arc<AtomicI32>,
    squelch_pending: Arc<AtomicI32>,
    #[cfg(feature = "zmq")]
    zmq_config: Option<(String, Option<String>, Option<String>)>,
    #[cfg(feature = "zmq")]
    hb_state_for_decode: Option<Arc<Mutex<bd_output::control::HeartbeatState>>>,
    #[cfg(feature = "gps")]
    gps_client: Option<bd_output::gps::GpsClient>,
) -> Result<(), String> {
    use std::sync::atomic::AtomicU64;

    const GPU_BATCH_SIZE: usize = 4096;

    let sdr = sdr;
    let max_samps = sdr.max_samps();

    let gpu = bd_gpu::GpuChannelizer::new(
        num_channels, semi_len, prototype, GPU_BATCH_SIZE,
    )?;

    let buffer_len = gpu.buffer_len();
    eprintln!("GPU: batch={} buffer={}KB result={}KB max_recv={}",
        GPU_BATCH_SIZE, buffer_len / 1024,
        (GPU_BATCH_SIZE * num_channels * 8) / 1024, max_samps);

    let overflow_count = Arc::new(AtomicU64::new(0));

    // Spawn parallel burst workers + decode thread
    let (batch_txs, worker_handles, decode_handle) = spawn_parallel_pipeline(
        num_channels,
        fft_scale, // GPU output is raw, workers apply fft_scale
        live_ch,
        first_live,
        last_live,
        burst_catchers,
        fsk,
        aa_correlator,
        aa_correlator_2m,
        syndrome_map,
        conn_table,
        pcap_writer,
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

    // SDR recv thread: continuously drains hardware, prevents overflow during GPU submit
    let (sdr_tx, sdr_rx) = channel::bounded::<Vec<i8>>(32);
    let sdr_overflow = overflow_count.clone();
    let sdr_running = running.clone();
    let sdr_buf_size = max_samps * 2;

    let sdr_gain_pending = gain_pending;
    let sdr_thread = std::thread::Builder::new()
        .name("sdr-recv-gpu".to_string())
        .spawn(move || {
            let mut sdr = sdr;
            let mut buf = vec![0i8; sdr_buf_size];
            while sdr_running.load(Ordering::Relaxed) {
                // Check for runtime gain change from C2
                let pending_gain = sdr_gain_pending.swap(i32::MIN, Ordering::Relaxed);
                if pending_gain != i32::MIN {
                    let gain_db = pending_gain as f64 / 10.0;
                    sdr.set_gain(gain_db, None, None);
                }

                let num_rx = sdr.recv_into(&mut buf);
                if num_rx == 0 {
                    continue;
                }
                sdr_overflow.store(sdr.overflow_count(), Ordering::Relaxed);
                let n = num_rx * 2;
                let mut send_buf = vec![0i8; n];
                send_buf.copy_from_slice(&buf[..n]);
                if sdr_tx.send(send_buf).is_err() {
                    break;
                }
            }
        })
        .expect("failed to spawn sdr-recv-gpu thread");

    let mut pos: usize = 0;
    let mut raw_buf = gpu.raw_buffer();

    for sc8_buf in sdr_rx.iter() {
        // Copy SC8 data into GPU raw buffer, handling partial fills
        let mut src_pos = 0usize;
        let n = sc8_buf.len();
        while src_pos < n {
            let copy_len = (n - src_pos).min(buffer_len - pos);
            // Safety: i8 and i8 have same layout
            raw_buf[pos..pos + copy_len].copy_from_slice(
                unsafe { std::slice::from_raw_parts(sc8_buf[src_pos..].as_ptr() as *const i8, copy_len) },
            );
            pos += copy_len;
            src_pos += copy_len;

            if pos >= buffer_len {
                if let Some(result) = gpu.submit() {
                    let ts = {
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default();
                        Timespec {
                            tv_sec: now.as_secs(),
                            tv_nsec: now.subsec_nanos() as u64,
                        }
                    };
                    broadcast_batch(&batch_txs, result.to_vec(), GPU_BATCH_SIZE, &ts);
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
            let ts = Timespec { tv_sec: 0, tv_nsec: 0 };
            broadcast_batch(&batch_txs, result.to_vec(), GPU_BATCH_SIZE, &ts);
        }
    }

    if let Some(result) = gpu.flush() {
        let ts = Timespec { tv_sec: 0, tv_nsec: 0 };
        broadcast_batch(&batch_txs, result.to_vec(), GPU_BATCH_SIZE, &ts);
    }

    drop(batch_txs);
    for h in worker_handles {
        let _ = h.join();
    }
    let _ = decode_handle.join();
    let _ = sdr_thread.join();

    Ok(())
}



