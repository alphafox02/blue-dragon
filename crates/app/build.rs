// Copyright 2025-2026 CEMAXECUTER LLC

fn main() {
    // Aaronia RTSA Suite: embed rpath so the binary finds libAaroniaRTSAAPI.so
    // at runtime without LD_LIBRARY_PATH.
    #[cfg(feature = "aaronia")]
    {
        let aaronia_lib_dir = "/opt/aaronia-rtsa-suite/Aaronia-RTSA-Suite-PRO";
        if std::path::Path::new(aaronia_lib_dir).is_dir() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", aaronia_lib_dir);
        }
    }
}
