fn main() {
    let csrc = std::path::PathBuf::from("csrc");

    cc::Build::new()
        .file(csrc.join("gpu_pfb_fft.c"))
        .include(&csrc) // for vkFFT.h
        .define("CL_TARGET_OPENCL_VERSION", "120")
        .define("VKFFT_BACKEND", "3")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-sign-compare")
        .flag("-Wno-deprecated-declarations")
        .flag("-O2")
        .compile("gpu_pfb_fft");

    println!("cargo:rustc-link-lib=OpenCL");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-changed=csrc/gpu_pfb_fft.c");
    println!("cargo:rerun-if-changed=csrc/vkFFT.h");
}
