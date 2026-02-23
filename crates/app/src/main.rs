mod pipeline;

use clap::Parser;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "blue-dragon")]
#[command(about = "Wideband BLE/Classic BT passive sniffer")]
struct Cli {
    /// Live capture mode
    #[arg(short = 'l', long, alias = "capture")]
    live: bool,

    /// IQ file input (for offline processing)
    #[arg(short = 'f', long)]
    file: Option<PathBuf>,

    /// SDR interface (e.g., usrp-B210-SERIAL)
    #[arg(short = 'i', long, alias = "extcap-interface")]
    interface: Option<String>,

    /// Sample format for file input: ci8, ci16, cf32
    #[arg(long, default_value = "ci16")]
    format: String,

    /// Center frequency in MHz
    #[arg(short = 'c', long, default_value = "2441")]
    center_freq: u32,

    /// Number of channels (bandwidth = channels * 1 MHz)
    #[arg(short = 'C', long, default_value = "40")]
    channels: usize,

    /// All channels: sets -C 96 -c 2441 for full BLE band coverage
    #[arg(short = 'a', long = "all-channels")]
    all_channels: bool,

    /// PCAP output file or FIFO
    #[arg(short = 'w', long, alias = "fifo")]
    write: Option<PathBuf>,

    /// Enable CRC checking
    #[arg(long)]
    check_crc: bool,

    /// SDR gain in dB
    #[arg(short = 'g', long, default_value = "60")]
    gain: f64,

    /// Squelch threshold in dB
    #[arg(short = 's', long, default_value = "-45")]
    squelch: f32,

    /// HackRF LNA gain (dB)
    #[arg(long, default_value = "40")]
    hackrf_lna: u32,

    /// HackRF VGA gain (dB)
    #[arg(long, default_value = "20")]
    hackrf_vga: u32,

    /// ZMQ endpoint (sensor connects PUB, e.g. tcp://collector:5555)
    #[arg(short = 'Z', long)]
    zmq: Option<String>,

    /// CurveZMQ keyfile for encryption
    #[arg(long)]
    zmq_curve_key: Option<String>,

    /// Sensor identifier for ZMQ/C2
    #[arg(long)]
    sensor_id: Option<String>,

    /// Enable gpsd GPS tagging
    #[arg(long)]
    gpsd: bool,

    /// Enable HCI GATT probing (opt-in active queries via system Bluetooth adapter)
    #[arg(long)]
    hci: bool,

    /// Verbose output
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Print statistics
    #[arg(long)]
    stats: bool,

    /// Enable continuous LE Coded PHY (Long Range) scan on advertising channels.
    /// Captures samples on channels 37/38/39 regardless of squelch, catching
    /// weak coded signals that don't trigger normal burst detection.
    /// Without this flag, coded decoding still runs on squelch-triggered bursts.
    #[arg(long)]
    coded_scan: bool,

    /// Disable GPU acceleration (use CPU-only PFB+FFT)
    #[arg(long)]
    no_gpu: bool,

    /// List available SDR interfaces
    #[arg(long)]
    list: bool,

    // -- Wireshark extcap args --

    /// Wireshark extcap: list interfaces
    #[arg(long)]
    extcap_interfaces: bool,

    /// Wireshark extcap: list DLTs
    #[arg(long)]
    extcap_dlts: bool,

    /// Wireshark extcap: show config
    #[arg(long)]
    extcap_config: bool,

    /// Wireshark extcap: version (ignored)
    #[arg(long, hide = true)]
    extcap_version: Option<String>,

    /// Install as Wireshark extcap plugin
    #[arg(long)]
    install: bool,
}

fn print_extcap_interfaces() {
    println!("extcap {{version=1.0}}");
    #[cfg(feature = "usrp")]
    {
        if let Ok(devices) = bd_sdr::usrp::list_devices() {
            for dev in &devices {
                let supported = if dev.device_type == "b200" { "" } else { " (unsupported)" };
                println!(
                    "interface {{value=usrp-{}-{}}}{{display=Blue Dragon{}}}",
                    dev.product, dev.serial, supported
                );
            }
        }
    }
}

fn print_extcap_dlts() {
    println!("dlt {{number=255}}{{name=LINKTYPE_BLUETOOTH_BREDR_BB}}{{display=Bluetooth BR/EDR and LE}}");
    println!("dlt {{number=256}}{{name=DLT_BLUETOOTH_LE_LL_WITH_PHDR}}{{display=Bluetooth LE}}");
}

fn print_extcap_config() {
    println!("arg {{number=0}}{{call=--channels}}{{display=Channels}}{{tooltip=Number of channels to capture}}{{type=selector}}");
    for i in (4..64).step_by(4) {
        println!("value {{arg=0}}{{value={}}}{{display={}}}{{default=false}}", i, i);
    }
    println!("value {{arg=0}}{{value=96}}{{display=96}}{{default=true}}");
    println!("arg {{number=1}}{{call=--center-freq}}{{display=Center Frequency}}{{tooltip=Center frequency to capture on}}{{type=integer}}{{range=2400,2480}}{{default=2441}}");
}

fn install_extcap() {
    let home = std::env::var("HOME").unwrap_or_else(|_| {
        eprintln!("error: $HOME not set");
        std::process::exit(1);
    });

    let extcap_dir = format!("{}/.config/wireshark/extcap", home);
    std::fs::create_dir_all(&extcap_dir).unwrap_or_else(|e| {
        eprintln!("error creating {}: {}", extcap_dir, e);
        std::process::exit(1);
    });

    let exe = std::env::current_exe().unwrap_or_else(|e| {
        eprintln!("error getting executable path: {}", e);
        std::process::exit(1);
    });

    let link_path = format!("{}/blue-dragon", extcap_dir);

    // Remove existing symlink if present
    let _ = std::fs::remove_file(&link_path);

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&exe, &link_path).unwrap_or_else(|e| {
            eprintln!("error creating symlink: {}", e);
            std::process::exit(1);
        });
    }

    eprintln!("Blue Dragon installed to Wireshark extcap directory");
    eprintln!("  {} -> {}", link_path, exe.display());
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Handle extcap commands first (Wireshark integration)
    if cli.extcap_interfaces {
        print_extcap_interfaces();
        return;
    }
    if cli.extcap_dlts {
        print_extcap_dlts();
        return;
    }
    if cli.extcap_config {
        print_extcap_config();
        return;
    }
    if cli.install {
        install_extcap();
        return;
    }

    if cli.list {
        let mut found = 0;
        #[cfg(feature = "usrp")]
        {
            match bd_sdr::usrp::list_devices() {
                Ok(devices) => {
                    for dev in &devices {
                        eprintln!("  usrp-{}-{} (type={})", dev.product, dev.serial, dev.device_type);
                        found += 1;
                    }
                }
                Err(e) => eprintln!("error listing USRP devices: {}", e),
            }
        }
        #[cfg(feature = "hackrf")]
        {
            match bd_sdr::hackrf::list_devices() {
                Ok(devices) => {
                    for dev in &devices {
                        eprintln!("  hackrf-{}", dev.serial);
                        found += 1;
                    }
                }
                Err(e) => eprintln!("error listing HackRF devices: {}", e),
            }
        }
        #[cfg(feature = "bladerf")]
        {
            match bd_sdr::bladerf::list_devices() {
                Ok(devices) => {
                    for dev in &devices {
                        eprintln!("  bladerf{} (serial={})", dev.instance, dev.serial);
                        found += 1;
                    }
                }
                Err(e) => eprintln!("error listing bladeRF devices: {}", e),
            }
        }
        #[cfg(feature = "soapysdr")]
        {
            match bd_sdr::soapysdr::list_devices() {
                Ok(devices) => {
                    for dev in &devices {
                        eprintln!("  soapy-{} (driver={})", dev.index, dev.driver);
                        found += 1;
                    }
                }
                Err(e) => eprintln!("error listing SoapySDR devices: {}", e),
            }
        }
        if found == 0 {
            eprintln!("  (no SDR devices found)");
        }
        return;
    }

    let (center_freq, channels) = if cli.all_channels {
        (2441u32, 96usize)
    } else {
        (cli.center_freq, cli.channels)
    };

    if cli.verbose {
        log::info!("blue-dragon starting");
        log::info!("center frequency: {} MHz", center_freq);
        log::info!("channels: {}", channels);
    }

    if let Some(ref file) = cli.file {
        let format = match cli.format.as_str() {
            "ci8" => bd_sdr::file::SampleFormat::Ci8,
            "ci16" => bd_sdr::file::SampleFormat::Ci16,
            "cf32" => bd_sdr::file::SampleFormat::Cf32,
            other => {
                eprintln!("unknown sample format: {} (use ci8, ci16, or cf32)", other);
                std::process::exit(1);
            }
        };

        if let Err(e) = pipeline::run_file(
            file,
            format,
            center_freq,
            channels,
            cli.write.as_deref(),
            cli.check_crc,
            cli.squelch,
            cli.stats,
        ) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    } else if cli.live {
        let iface = cli.interface.as_deref().unwrap_or_else(|| {
            eprintln!("error: live mode requires -i <interface> (e.g., -i usrp-B210-SERIAL)");
            eprintln!("use --list to see available interfaces");
            std::process::exit(1);
        });

        // Set up Ctrl-C handler
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();
        ctrlc::set_handler(move || {
            eprintln!("\ninterrupted, stopping...");
            r.store(false, Ordering::SeqCst);
        })
        .expect("failed to set Ctrl-C handler");

        let use_gpu = cfg!(feature = "gpu") && !cli.no_gpu;

        // Auto-generate sensor_id from hostname when using ZMQ but no explicit ID
        let sensor_id = cli.sensor_id.clone().or_else(|| {
            if cli.zmq.is_some() {
                std::fs::read_to_string("/etc/hostname")
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            }
        });

        if let Err(e) = pipeline::run_live(
            iface,
            center_freq,
            channels,
            cli.gain,
            cli.squelch,
            cli.hackrf_lna,
            cli.hackrf_vga,
            cli.write.as_deref(),
            cli.check_crc,
            cli.stats,
            use_gpu,
            cli.zmq.as_deref(),
            cli.zmq_curve_key.as_deref(),
            sensor_id.as_deref(),
            cli.gpsd,
            cli.hci,
            cli.coded_scan,
            running,
        ) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    } else {
        eprintln!("no input specified. Use -f <file> for file input or -l for live SDR.");
        std::process::exit(1);
    }
}
