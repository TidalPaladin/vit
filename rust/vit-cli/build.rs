//! Build script for vit-cli.
//!
//! Sets RPATH so the binary can find libraries relative to itself.

fn main() {
    // Only set rpath when building with FFI feature
    #[cfg(feature = "ffi")]
    {
        // Use $ORIGIN to look for libraries relative to the binary
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");

        // Also add libtorch path if available
        if let Ok(libtorch) = std::env::var("LIBTORCH") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", libtorch);
        }
    }
}
