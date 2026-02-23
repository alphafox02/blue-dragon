// Copyright 2025-2026 CEMAXECUTER LLC

fn main() {
    let csrc = std::path::PathBuf::from("csrc");

    let mut build = cc::Build::new();
    build
        .file(csrc.join("gpu_pfb_fft.c"))
        .include(&csrc) // for vkFFT.h
        .define("CL_TARGET_OPENCL_VERSION", "120")
        .define("VKFFT_BACKEND", "3")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-sign-compare")
        .flag("-Wno-deprecated-declarations")
        .flag("-O2");

    // macOS: OpenCL is a framework, not a library
    if cfg!(target_os = "macos") {
        build.flag("-framework").flag("OpenCL");
    }

    build.compile("gpu_pfb_fft");

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
    } else {
        println!("cargo:rustc-link-lib=OpenCL");
    }
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-changed=csrc/gpu_pfb_fft.c");
    println!("cargo:rerun-if-changed=csrc/vkFFT.h");
}
