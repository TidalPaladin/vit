//! Build script for vit-ffi.
//!
//! This script builds the C++ bridge library and links it to the Rust crate.
//!
//! # Environment Variables
//!
//! - `LIBTORCH`: Path to libtorch installation (required)
//! - `LIBTORCH_CXX11_ABI`: Set to "1" to use the CXX11 ABI (default: "0")
//! - `VIT_BRIDGE_SKIP_BUILD`: Set to "1" to skip building (for development)
//! - `ROCM_PATH`: Path to ROCm installation (optional, for ROCm builds)

use std::env;
use std::path::PathBuf;

fn main() {
    // Check if we should skip the build
    if env::var("VIT_BRIDGE_SKIP_BUILD").map(|v| v == "1").unwrap_or(false) {
        println!("cargo:warning=Skipping vit-bridge build (VIT_BRIDGE_SKIP_BUILD=1)");
        return;
    }

    // Get libtorch path
    let libtorch = match env::var("LIBTORCH") {
        Ok(path) => PathBuf::from(path),
        Err(_) => {
            println!("cargo:warning=LIBTORCH environment variable not set");
            println!("cargo:warning=Set LIBTORCH to your libtorch installation path");
            println!("cargo:warning=Example: export LIBTORCH=/path/to/libtorch");
            println!("cargo:warning=Skipping C++ bridge build");
            return;
        }
    };

    if !libtorch.exists() {
        println!("cargo:warning=LIBTORCH path does not exist: {}", libtorch.display());
        println!("cargo:warning=Skipping C++ bridge build");
        return;
    }

    // Rerun if bridge sources change
    println!("cargo:rerun-if-changed=../bridge/src/vit_bridge.cpp");
    println!("cargo:rerun-if-changed=../bridge/include/vit_bridge.h");
    println!("cargo:rerun-if-changed=../bridge/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    // Get the bridge directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bridge_dir = manifest_dir.parent().unwrap().join("bridge");

    // Build with CMake
    let mut cmake_config = cmake::Config::new(&bridge_dir);

    // Set libtorch path
    cmake_config.define("CMAKE_PREFIX_PATH", &libtorch);

    // Set build type
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let build_type = if profile == "release" {
        "Release"
    } else {
        "Debug"
    };
    cmake_config.define("CMAKE_BUILD_TYPE", build_type);

    // Handle CXX11 ABI
    let cxx11_abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "0".to_string());
    cmake_config.define("CMAKE_CXX_FLAGS", format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi));

    // Check for CUDA feature
    if cfg!(feature = "cuda") {
        cmake_config.define("USE_CUDA", "ON");
    }

    // Check for ROCm feature
    if cfg!(feature = "rocm") {
        cmake_config.define("USE_HIP", "ON");
        // Pass ROCM_PATH to CMake if set
        if let Ok(rocm_path) = env::var("ROCM_PATH") {
            cmake_config.define("ROCM_PATH", rocm_path);
        }
    }

    // Check if libtorch has CUDA support by looking for CUDA cmake files
    let libtorch_has_cuda = libtorch.join("share/cmake/Caffe2/public/cuda.cmake").exists();
    // Check if libtorch has HIP/ROCm support
    let libtorch_has_hip = libtorch.join("share/cmake/Caffe2/public/LoadHIP.cmake").exists()
        || libtorch.join("share/cmake/Caffe2/Caffe2HIPConfig.cmake").exists();

    // Set CUDA architectures if libtorch has CUDA support (not ROCm)
    if libtorch_has_cuda && !libtorch_has_hip && std::path::Path::new("/usr/local/cuda").exists() {
        cmake_config.define("CMAKE_CUDA_ARCHITECTURES", "native");

        // Use GCC 13 if available (CUDA 12.x doesn't support GCC 14+)
        // Note: ROCm supports newer GCC versions, so this constraint only applies to CUDA
        if std::path::Path::new("/usr/bin/gcc-13").exists() {
            cmake_config.define("CMAKE_C_COMPILER", "/usr/bin/gcc-13");
            cmake_config.define("CMAKE_CXX_COMPILER", "/usr/bin/g++-13");
            cmake_config.define("CMAKE_CUDA_HOST_COMPILER", "/usr/bin/g++-13");
        }
    }

    // Build
    let dst = cmake_config.build();

    // Link the bridge library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=vit_bridge");

    // Link libtorch libraries
    let lib_dir = libtorch.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    // Link CUDA libraries if available
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
    }

    // Link ROCm/HIP libraries if available
    if cfg!(feature = "rocm") {
        println!("cargo:rustc-link-lib=dylib=torch_hip");
        println!("cargo:rustc-link-lib=dylib=c10_hip");
    }

    // Set rpath for runtime library loading
    // Use $ORIGIN for portable binary (looks in ./lib relative to binary)
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
    // Also add absolute paths as fallback
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", dst.display());
}
