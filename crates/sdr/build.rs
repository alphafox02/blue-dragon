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
    }
}
