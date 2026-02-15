use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use crossbeam::channel;

use bt_dsp::burst::BurstCatcher;
use bt_dsp::fft::BatchFft;
use bt_dsp::fsk::FskDemod;
use bt_dsp::pfb::PfbChannelizer;
use bt_dsp::window;
use bt_output::pcap::PcapWriter;
use bt_protocol::ble::{self, AaCorrelator};
use bt_protocol::ble_connection::ConnectionTable;
use bt_protocol::btbb::{self, SyndromeMap};
use bt_protocol::Timespec;
use bt_sdr::file::{FileSource, SampleFormat};
use bt_sdr::SdrSource;

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

    // Build channel frequency table
    let channel_freqs: Vec<u32> = (0..num_channels)
        .map(|i| {
            let offset = i as i32 - (num_channels as i32) / 2;
            (center_freq_mhz as i32 + offset) as u32
        })
        .collect();

    log::info!("channels: {} ({}-{} MHz)", num_channels,
               channel_freqs.first().unwrap_or(&0),
               channel_freqs.last().unwrap_or(&0));

    // Initialize protocol subsystems
    let aa_correlator = AaCorrelator::new();
    let syndrome_map = SyndromeMap::new(1);
    let mut conn_table = ConnectionTable::new();

    // Initialize DSP
    let prototype = window::pfb_prototype(num_channels, 12, 7.0);
    let mut channelizer = PfbChannelizer::new(num_channels, &prototype);
    let mut fft = BatchFft::new(num_channels);
    let sps = 2usize; // samples per symbol for BLE

    // Per-channel burst catchers
    let mut burst_catchers: Vec<BurstCatcher> = channel_freqs
        .iter()
        .map(|&freq| BurstCatcher::new(freq, squelch_db))
        .collect();

    // FSK demodulator (one per thread, but we're single-threaded for file input)
    let mut fsk = FskDemod::new(sps);

    // PCAP writer
    let mut pcap_writer: Option<PcapWriter<BufWriter<File>>> = if let Some(path) = pcap_path {
        let file = File::create(path).map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
        let writer = BufWriter::new(file);
        Some(PcapWriter::new(writer).map_err(|e| format!("failed to write PCAP header: {}", e))?)
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

    // File source (blocking read, single-threaded for Phase 1)
    let mut source = FileSource::new(
        file_path.to_string_lossy().to_string(),
        format,
        sample_rate,
        center_freq_hz,
    );

    let (tx, rx) = channel::bounded(64);

    // Start file reader in a separate thread
    let reader_thread = std::thread::spawn(move || {
        if let Err(e) = source.start(tx) {
            log::error!("file reader error: {}", e);
        }
    });

    // Timestamp counter (sample-based, converted to timespec)
    let mut sample_count: u64 = 0;
    let samples_to_timespec = |count: u64, rate: u32| -> Timespec {
        let secs = count / rate as u64;
        let frac = count % rate as u64;
        let nsec = frac * 1_000_000_000 / rate as u64;
        Timespec { tv_sec: secs, tv_nsec: nsec }
    };

    // Main processing loop
    for buf in rx.iter() {
        // Process M samples at a time through channelizer
        let num_blocks = buf.num_samples / num_channels;

        for block in 0..num_blocks {
            let offset = block * num_channels * 2;
            let end = offset + num_channels * 2;
            if end > buf.data.len() {
                break;
            }

            channelizer.push_samples(&buf.data[offset..end]);
            let pfb_out = channelizer.filter_outputs();

            // FFT the polyphase outputs
            let mut fft_buf = pfb_out;
            fft.process(&mut fft_buf);

            let ts = samples_to_timespec(sample_count, sample_rate);

            // Feed each channel through its burst catcher
            for (ch, &sample) in fft_buf.iter().enumerate() {
                if ch >= num_channels {
                    break;
                }
                if let Some(burst) = burst_catchers[ch].execute(sample, &ts) {
                    total_bursts += 1;

                    // FSK demodulate the burst
                    if let Some(fsk_result) = fsk.demodulate(&burst.samples) {
                        let freq = channel_freqs[ch];
                        let burst_ts = burst.timestamp.clone();
                        let rssi = burst.rssi_db as i32;
                        let noise = burst.noise_db as i32;

                        // Try Classic BT first
                        if let Some(bt_pkt) = btbb::detect(
                            &fsk_result.bits, freq, rssi, noise,
                            burst_ts.clone(), &syndrome_map,
                        ) {
                            total_bt += 1;
                            if let Some(ref mut writer) = pcap_writer {
                                let _ = writer.write_bt(&bt_pkt, None);
                            }
                        } else {
                            // Try BLE preamble-first detection
                            let mut pkt = ble::ble_burst(
                                &fsk_result.bits, freq, burst_ts.clone(),
                                check_crc,
                                |aa| conn_table.crc_init_for_aa(aa, burst_ts.tv_sec),
                            );

                            // Fall back to AA correlator
                            if pkt.is_none() {
                                pkt = aa_correlator.correlate(
                                    &fsk_result.demod, freq, burst_ts.clone(), check_crc,
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

            sample_count += num_channels as u64;
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
                elapsed, total_ble, total_bt, total_bursts,
                crc_pct, valid_crc, total_crc, conns,
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
            elapsed, total_ble, total_bt, total_bursts,
            crc_pct, valid_crc, total_crc,
        );
    }

    // Wait for reader thread
    let _ = reader_thread.join();

    Ok(())
}
