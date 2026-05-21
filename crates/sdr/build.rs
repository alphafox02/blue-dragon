// Copyright 2025-2026 CEMAXECUTER LLC

use std::path::Path;
use std::process::Command;

/// Try pkg-config for a library. Returns true if found.
fn try_pkg_config(lib: &str) -> bool {
    Command::new("pkg-config")
        .args(["--libs", lib])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Search common library directories for source-built libraries.
/// DragonOS and similar distros often install SDR libs from source
/// into /usr/local/{lib,include} or /opt/... paths.
fn add_search_paths() {
    let lib_dirs = [
        "/usr/local/lib",
        "/usr/local/lib/x86_64-linux-gnu",
        "/usr/local/lib/aarch64-linux-gnu",
    ];
    let include_dirs = [
        "/usr/local/include",
    ];

    for dir in &lib_dirs {
        if Path::new(dir).is_dir() {
            println!("cargo:rustc-link-search=native={}", dir);
        }
    }
    for dir in &include_dirs {
        if Path::new(dir).is_dir() {
            println!("cargo:include={}", dir);
        }
    }
}

/// Link a library, preferring pkg-config but falling back to direct link.
fn link_lib(pkg_name: &str, lib_name: &str) {
    if try_pkg_config(pkg_name) {
        // pkg-config found it -- emit the flags it provides
        let output = Command::new("pkg-config")
            .args(["--libs", pkg_name])
            .output()
            .unwrap();
        let flags = String::from_utf8_lossy(&output.stdout);
        for flag in flags.split_whitespace() {
            if let Some(dir) = flag.strip_prefix("-L") {
                println!("cargo:rustc-link-search=native={}", dir);
            } else if let Some(lib) = flag.strip_prefix("-l") {
                println!("cargo:rustc-link-lib={}", lib);
            }
        }
    } else {
        // No pkg-config -- link directly (search paths already added)
        println!("cargo:rustc-link-lib={}", lib_name);
    }
}

fn main() {
    add_search_paths();

    // Compile WHAD protobuf definitions
    let proto_root = "proto";
    let proto_files = [
        "whad/protocol/whad.proto",
        "whad/protocol/device.proto",
        "whad/protocol/generic.proto",
        "whad/protocol/ble/ble.proto",
    ];
    let proto_paths: Vec<String> = proto_files.iter().map(|f| format!("{}/{}", proto_root, f)).collect();
    if proto_paths.iter().all(|p| Path::new(p).exists()) {
        prost_build::Config::new()
            .out_dir("src/whad_proto")
            .compile_protos(
                &proto_paths.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                &[proto_root],
            )
            .expect("failed to compile WHAD proto files");
    }

    #[cfg(feature = "usrp")]
    link_lib("uhd", "uhd");

    #[cfg(feature = "hackrf")]
    link_lib("libhackrf", "hackrf");

    #[cfg(feature = "bladerf")]
    {
        // bladeRF often lacks a .pc file when built from source
        if try_pkg_config("bladeRF") {
            link_lib("bladeRF", "bladeRF");
        } else {
            println!("cargo:rustc-link-lib=bladeRF");
        }
    }

    #[cfg(feature = "aaronia")]
    {
        // Aaronia RTSA Suite installs to /opt/aaronia-rtsa-suite/
        let aaronia_lib_dir = "/opt/aaronia-rtsa-suite/Aaronia-RTSA-Suite-PRO";
        if Path::new(aaronia_lib_dir).is_dir() {
            println!("cargo:rustc-link-search=native={}", aaronia_lib_dir);
        }
        println!("cargo:rustc-link-lib=AaroniaRTSAAPI");
    }

    #[cfg(feature = "rfnm")]
    {
        // Compile C++ shim for librfnm
        println!("cargo:rerun-if-changed=csrc/rfnm_shim.cpp");
        println!("cargo:rerun-if-changed=csrc/rfnm_shim.h");
        let mut build = cc::Build::new();
        build.cpp(true);
        build.file("csrc/rfnm_shim.cpp");
        build.include("/usr/local/include");
        build.flag("-std=c++17");
        // Match librfnm's spdlog defines so our set_level() operates on the
        // same shared default logger instance, not a header-only copy.
        build.define("SPDLOG_COMPILED_LIB", None);
        build.define("SPDLOG_FMT_EXTERNAL", None);
        build.define("SPDLOG_SHARED_LIB", None);
        build.define("FMT_SHARED", None);
        build.compile("rfnm_shim");

        link_lib("librfnm", "rfnm");
        println!("cargo:rustc-link-lib=usb-1.0");
        println!("cargo:rustc-link-lib=stdc++");
        // spdlog used by librfnm; we call set_level to suppress noisy internal logging
        println!("cargo:rustc-link-lib=spdlog");
        println!("cargo:rustc-link-lib=fmt");
    }

    #[cfg(feature = "sidekiq")]
    {
        configure_sidekiq();
    }

    #[cfg(feature = "soapysdr")]
    {
        link_lib("SoapySDR", "SoapySDR");

        // Compile C shim for SoapySDR (works around FFI ABI issue with SoapyUHD)
        let mut build = cc::Build::new();
        build.file("csrc/soapy_shim.c");

        // Add include paths where SoapySDR headers might live
        for dir in &["/usr/include", "/usr/local/include"] {
            if Path::new(dir).join("SoapySDR").is_dir() {
                build.include(dir);
            }
        }
        build.compile("soapy_shim");
    }
}

/// Locate the Epiq Sidekiq SDK and emit link directives for it.
///
/// Mirrors the auto-detection logic in the SDR++ sidekiq_source CMakeLists.txt:
///
/// 1. SDK root: `$Sidekiq_DIR` || `$SIDEKIQ_ROOT` || `$HOME/sidekiq_sdk_current`.
/// 2. Archive suffix: `$SIDEKIQ_SUFFIX` override, else host-arch-preferred list,
///    else any archive that also has a matching `lib/support/<suffix>/` dir.
/// 3. Per-suffix extras: Z4 uses a shared `.so` + bundled libgpiod; X40 pulls in
///    grpc++/grpc/gpr/protobuf/absl/cares/ssl/crypto + libgpiod; z3u/aarch64/
///    arm-cortex SDKs require libiio.
/// 4. Common deps: glib-2.0, libusb-1.0, libtirpc, pthread, dl, rt, m, stdc++.
/// 5. RPATH: SDK support bundle + `/usr/lib/epiq` if installed.
#[cfg(feature = "sidekiq")]
fn configure_sidekiq() {
    println!("cargo:rerun-if-env-changed=Sidekiq_DIR");
    println!("cargo:rerun-if-env-changed=SIDEKIQ_ROOT");
    println!("cargo:rerun-if-env-changed=SIDEKIQ_SUFFIX");

    let sdk_root = std::env::var("Sidekiq_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("SIDEKIQ_ROOT").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{}/sidekiq_sdk_current", home)
        });

    if !Path::new(&sdk_root).is_dir() {
        panic!(
            "Sidekiq SDK not found at '{}'. Set Sidekiq_DIR/SIDEKIQ_ROOT or \
             install the SDK to ~/sidekiq_sdk_current.\n\
             Download: https://epiq-solutions.com/sidekiq/",
            sdk_root
        );
    }

    let header = format!("{}/sidekiq_core/inc/sidekiq_api.h", sdk_root);
    if !Path::new(&header).exists() {
        panic!(
            "Sidekiq headers not found at {} (Sidekiq_DIR={})",
            header, sdk_root
        );
    }

    // ---- Pick archive suffix ----
    let suffix = std::env::var("SIDEKIQ_SUFFIX")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| pick_sidekiq_suffix(&sdk_root))
        .unwrap_or_else(|| {
            panic!(
                "Could not determine Sidekiq SDK suffix. Checked archives in \
                 {}/lib. Set SIDEKIQ_SUFFIX (e.g. x86_64.gcc, aarch64.gcc6.3, \
                 z3u, z4, msiq-x40).",
                sdk_root
            )
        });

    let arch_type = suffix.split('.').next().unwrap_or(&suffix).to_string();
    let support_dir = format!("{}/lib/support/{}/usr/lib/epiq", sdk_root, suffix);
    let support_pkgconfig = format!("{}/pkgconfig", support_dir);

    println!("cargo:warning=sidekiq: SDK root = {}", sdk_root);
    println!("cargo:warning=sidekiq: archive suffix = {}", suffix);
    println!("cargo:warning=sidekiq: arch type = {}", arch_type);

    // ---- Z4: shared .so override + libgpiod from SDK bundle ----
    if suffix == "z4" {
        let so_candidates = [
            format!("{}/libsidekiq-dev-z4-1.so", support_dir),
            format!("{}/libsidekiq.so", support_dir),
        ];
        let so_found = so_candidates.iter().find(|p| Path::new(p).exists());
        if let Some(so) = so_found {
            // Use absolute path linkage so we don't have to worry about name suffixes.
            println!("cargo:rustc-link-search=native={}", support_dir);
            // Strip "lib" prefix and ".so" / ".so.*" suffix for -l name.
            let name = Path::new(so)
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_prefix("lib"))
                .and_then(|n| n.split(".so").next())
                .unwrap_or("sidekiq");
            println!("cargo:rustc-link-lib=dylib={}", name);
        } else {
            panic!("Z4: shared libsidekiq .so not found in {}", support_dir);
        }

        let gpiod = format!("{}/libgpiod.so", support_dir);
        if Path::new(&gpiod).exists() {
            println!("cargo:rustc-link-lib=dylib=gpiod");
        } else {
            panic!("Z4: libgpiod not found in {}", support_dir);
        }

        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", support_dir);
    } else {
        // Static archive linkage for every other suffix.
        let archive = format!("{}/lib/libsidekiq__{}.a", sdk_root, suffix);
        if !Path::new(&archive).exists() {
            panic!("libsidekiq__{}.a not found at {}", suffix, archive);
        }
        println!("cargo:rustc-link-search=native={}/lib", sdk_root);
        println!("cargo:rustc-link-lib=static=sidekiq__{}", suffix);
    }

    // ---- X40 (msiq-x40): gRPC + bundled deps + libgpiod ----
    if suffix == "msiq-x40" {
        if Path::new(&support_dir).is_dir() {
            println!("cargo:rustc-link-search=native={}", support_dir);
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", support_dir);
        }
        for lib in &[
            "grpc++",
            "grpc",
            "gpr",
            "protobuf",
            "address_sorting",
            "re2",
            "upb",
            "cares",
            "ssl",
            "crypto",
            "gpiod",
        ] {
            println!("cargo:rustc-link-lib=dylib={}", lib);
        }
        // absl_* libs vary by SDK version; glob them in.
        if let Ok(entries) = std::fs::read_dir(&support_dir) {
            for ent in entries.flatten() {
                let name = ent.file_name();
                let s = name.to_string_lossy();
                if let Some(stripped) = s.strip_prefix("libabsl_") {
                    if let Some(short) = stripped.split(".so").next() {
                        println!("cargo:rustc-link-lib=dylib=absl_{}", short);
                    }
                }
            }
        }
    }

    // ---- libiio (z3u + ARM SDKs) ----
    let needs_iio = matches!(
        suffix.as_str(),
        "z3u" | "aarch64" | "aarch64.gcc6.3" | "arm_cortex-a9.gcc7.2.1_gnueabihf"
    );
    if needs_iio {
        println!("cargo:rustc-link-lib=dylib=iio");
    }

    // ---- Pre-installed /usr/lib/epiq (Epiq's sidekiq-shared-libs-* deb) ----
    let epiq_lib = "/usr/lib/epiq";
    if Path::new(epiq_lib).is_dir() {
        println!("cargo:rustc-link-search=native={}", epiq_lib);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", epiq_lib);
    }

    // ---- SDK support bundle (.so deps for static-archive flavors) ----
    if suffix != "z4" && Path::new(&support_dir).is_dir() {
        println!("cargo:rustc-link-search=native={}", support_dir);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", support_dir);
    }

    // ---- pkg-config priming for glib / tirpc / libusb ----
    if Path::new(&support_pkgconfig).is_dir() {
        // Best-effort: prepend the support pkgconfig dir for pkg-config calls.
        let prev = std::env::var("PKG_CONFIG_PATH").unwrap_or_default();
        let combined = if prev.is_empty() {
            support_pkgconfig.clone()
        } else {
            format!("{}:{}", support_pkgconfig, prev)
        };
        std::env::set_var("PKG_CONFIG_PATH", &combined);
    }

    // ---- Common runtime deps (always needed by libsidekiq) ----
    for lib in &[
        "glib-2.0",
        "gobject-2.0",
        "usb-1.0",
        "tirpc",
        "pthread",
        "dl",
        "rt",
        "m",
        "stdc++",
    ] {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }
}

/// Mirror of SDR++ CMake's two-pass suffix detection:
/// 1) host-arch-preferred archive in `<root>/lib/libsidekiq__<sfx>.a`
/// 2) any archive whose suffix also has a matching `<root>/lib/support/<sfx>/` dir.
#[cfg(feature = "sidekiq")]
fn pick_sidekiq_suffix(sdk_root: &str) -> Option<String> {
    let lib_dir = format!("{}/lib", sdk_root);
    let preferred: &[&str] = if cfg!(target_arch = "x86_64") {
        &["x86_64.gcc", "x86_64", "z4"]
    } else if cfg!(target_arch = "aarch64") {
        &[
            "aarch64.gcc6.3",
            "aarch64",
            "z4",
            "arm_cortex-a9.gcc7.2.1_gnueabihf",
            "z3u",
        ]
    } else {
        &[
            "x86_64.gcc",
            "x86_64",
            "aarch64.gcc6.3",
            "aarch64",
            "z4",
            "z3u",
        ]
    };

    for sfx in preferred {
        let p = format!("{}/libsidekiq__{}.a", lib_dir, sfx);
        if Path::new(&p).exists() {
            return Some((*sfx).to_string());
        }
    }

    // Fallback: any archive that has a matching support bundle.
    if let Ok(entries) = std::fs::read_dir(&lib_dir) {
        for ent in entries.flatten() {
            let name = ent.file_name();
            let s = name.to_string_lossy();
            if let Some(stripped) = s.strip_prefix("libsidekiq__") {
                if let Some(sfx) = stripped.strip_suffix(".a") {
                    let support = format!("{}/support/{}", lib_dir, sfx);
                    if Path::new(&support).is_dir() {
                        return Some(sfx.to_string());
                    }
                }
            }
        }
    }
    None
}
