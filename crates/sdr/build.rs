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
