// Copyright 2025-2026 CEMAXECUTER LLC

fn main() {
    let csrc = std::path::PathBuf::from("csrc");

    let mut build = cc::Build::new();

    if cfg!(target_os = "macos") {
        // Metal backend: VkFFT backend 5 + Metal compute shader for PFB
        build
            .cpp(true)
            .file(csrc.join("gpu_pfb_fft_metal.cc"))
            .include(&csrc) // for vkFFT.h
            .include(csrc.join("metal-cpp")) // for Metal/Metal.hpp etc.
            .define("VKFFT_BACKEND", "5")
            .flag("-std=c++17")
            .flag("-Wno-unused-parameter")
            .flag("-Wno-sign-compare")
            .flag("-Wno-deprecated-declarations")
            .flag("-O2");
        build.compile("gpu_pfb_fft");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rerun-if-changed=csrc/gpu_pfb_fft_metal.cc");
    } else {
        // OpenCL backend: VkFFT backend 3 + OpenCL compute kernel for PFB
        build
            .file(csrc.join("gpu_pfb_fft.c"))
            .include(&csrc) // for vkFFT.h
            .define("CL_TARGET_OPENCL_VERSION", "120")
            .define("VKFFT_BACKEND", "3")
            .flag("-Wno-unused-parameter")
            .flag("-Wno-sign-compare")
            .flag("-Wno-deprecated-declarations")
            .flag("-O2");
        build.compile("gpu_pfb_fft");

        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rerun-if-changed=csrc/gpu_pfb_fft.c");
    }

    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-changed=csrc/vkFFT.h");
}
