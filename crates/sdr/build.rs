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
