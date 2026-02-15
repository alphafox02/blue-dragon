mod pipeline;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "bluetooth-sniffer")]
#[command(about = "Wideband BLE/Classic BT passive sniffer")]
struct Cli {
    /// IQ file input (for offline processing)
    #[arg(short = 'f', long)]
    file: Option<PathBuf>,

    /// Sample format for file input: ci8, ci16, cf32
    #[arg(long, default_value = "ci16")]
    format: String,

    /// Center frequency in MHz
    #[arg(short = 'c', long, default_value = "2441")]
    center_freq: u32,

    /// Number of channels (bandwidth = channels * 1 MHz)
    #[arg(short = 'C', long, default_value = "40")]
    channels: usize,

    /// PCAP output file
    #[arg(short = 'w', long)]
    write: Option<PathBuf>,

    /// Enable CRC checking
    #[arg(long)]
    check_crc: bool,

    /// Squelch threshold in dB
    #[arg(short = 's', long, default_value = "-45")]
    squelch: f32,

    /// Verbose output
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Print statistics
    #[arg(long)]
    stats: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    if cli.verbose {
        log::info!("bluetooth-sniffer starting");
        log::info!("center frequency: {} MHz", cli.center_freq);
        log::info!("channels: {}", cli.channels);
    }

    if let Some(ref file) = cli.file {
        let format = match cli.format.as_str() {
            "ci8" => bt_sdr::file::SampleFormat::Ci8,
            "ci16" => bt_sdr::file::SampleFormat::Ci16,
            "cf32" => bt_sdr::file::SampleFormat::Cf32,
            other => {
                eprintln!("unknown sample format: {} (use ci8, ci16, or cf32)", other);
                std::process::exit(1);
            }
        };

        if let Err(e) = pipeline::run_file(
            file,
            format,
            cli.center_freq,
            cli.channels,
            cli.write.as_deref(),
            cli.check_crc,
            cli.squelch,
            cli.stats,
        ) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    } else {
        eprintln!("no input specified. Use -f <file> for file input or -l for live SDR.");
        eprintln!("live SDR support coming in Phase 2.");
        std::process::exit(1);
    }
}
