fn main() {
    // Search /usr/local/lib for libraries installed from source
    println!("cargo:rustc-link-search=native=/usr/local/lib");

    #[cfg(feature = "usrp")]
    {
        println!("cargo:rustc-link-lib=uhd");
    }

    #[cfg(feature = "hackrf")]
    {
        println!("cargo:rustc-link-lib=hackrf");
    }

    #[cfg(feature = "bladerf")]
    {
        println!("cargo:rustc-link-lib=bladeRF");
    }

    #[cfg(feature = "soapysdr")]
    {
        println!("cargo:rustc-link-lib=SoapySDR");

        // Compile C shim for SoapySDR (works around FFI ABI issue with SoapyUHD)
        cc::Build::new()
            .file("csrc/soapy_shim.c")
            .include("/usr/include")
            .compile("soapy_shim");
    }
}
