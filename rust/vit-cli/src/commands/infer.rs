//! Infer command implementation.

use std::path::Path;

use anyhow::{bail, Result};

/// Run the infer command.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model_path: &Path,
    config_path: &Path,
    device_str: &str,
    dtype: &str,
    shape_str: &str,
    #[cfg_attr(not(feature = "ffi"), allow(unused_variables))] warmup: usize,
    #[cfg_attr(not(feature = "ffi"), allow(unused_variables))] iterations: usize,
) -> Result<()> {
    // Try to use the FFI implementation if available
    #[cfg(feature = "ffi")]
    {
        return run_with_ffi(
            model_path,
            config_path,
            device_str,
            dtype,
            shape_str,
            warmup,
            iterations,
        );
    }

    // FFI not available - show instructions
    #[cfg(not(feature = "ffi"))]
    {
        run_without_ffi(model_path, config_path, device_str, dtype, shape_str)
    }
}

#[cfg(feature = "ffi")]
fn run_with_ffi(
    model_path: &Path,
    config_path: &Path,
    device_str: &str,
    dtype: &str,
    shape_str: &str,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    use anyhow::Context;
    use vit_ffi::{Device, Model, Tensor};

    // Parse shape
    let shape: Vec<i64> = shape_str
        .split(',')
        .map(|s| s.trim().parse::<i64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("Invalid shape '{}'", shape_str))?;

    if shape.len() != 4 {
        bail!(
            "Expected 4D shape (batch, channels, height, width), got {}D",
            shape.len()
        );
    }

    // Parse device
    let device = Device::parse(device_str)
        .map_err(|e| anyhow::anyhow!("Invalid device '{}': {}", device_str, e))?;

    // Validate dtype
    if dtype != "float32" && dtype != "bfloat16" && dtype != "float16" {
        bail!(
            "Unsupported dtype '{}'. Use float32, float16, or bfloat16",
            dtype
        );
    }

    println!("Inference Configuration:");
    println!("  Model: {:?}", model_path);
    println!("  Config: {:?}", config_path);
    println!("  Device: {}", device);
    println!("  DType: {}", dtype);
    println!("  Shape: {:?}", shape);
    println!("  Warmup: {}", warmup);
    println!("  Iterations: {}", iterations);
    println!();

    // Check if model file exists
    if !model_path.exists() {
        bail!(
            "Model file not found: {:?}\n\n\
             To export a model, run:\n\n\
             python scripts/export_aot.py \\\n\
                 --config {:?} \\\n\
                 --output {:?} \\\n\
                 --shape {} \\\n\
                 --device {}",
            model_path,
            config_path,
            model_path,
            shape_str,
            device_str
        );
    }

    // Load the model
    println!("Loading model...");
    let model = Model::load(model_path, device)
        .with_context(|| format!("Failed to load model from {:?}", model_path))?;
    println!("Model loaded on {}", model.device());

    // Create random input tensor
    println!("Creating input tensor with shape {:?}...", shape);
    let input =
        Tensor::randn(&shape, device).with_context(|| "Failed to create input tensor")?;

    // Warmup iterations
    if warmup > 0 {
        println!("Running {} warmup iteration(s)...", warmup);
        for i in 0..warmup {
            let result = model
                .infer(&input)
                .with_context(|| format!("Warmup iteration {} failed", i + 1))?;
            // Sync to ensure completion
            if let Device::Cuda(idx) = device {
                vit_ffi::cuda_synchronize(idx);
            }
            drop(result);
        }
    }

    // Timed iterations
    println!("Running {} timed iteration(s)...", iterations);
    let mut latencies = Vec::with_capacity(iterations);
    let mut peak_memory: usize = 0;
    let mut output_shape = vec![];

    for i in 0..iterations {
        let result = model
            .infer(&input)
            .with_context(|| format!("Inference iteration {} failed", i + 1))?;

        latencies.push(result.latency_ms());
        peak_memory = peak_memory.max(result.memory_bytes());

        if i == 0 {
            output_shape = result.output().shape();
        }
    }

    // Compute statistics
    let mean_latency: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let std_latency: f64 = if latencies.len() > 1 {
        let variance: f64 = latencies
            .iter()
            .map(|x| (x - mean_latency).powi(2))
            .sum::<f64>()
            / (latencies.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let min_latency = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_latency = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Print results
    println!();
    println!("Results:");
    println!("  Output shape: {:?}", output_shape);
    println!("  Latency:");
    println!("    Mean:   {:.3} ms", mean_latency);
    println!("    Std:    {:.3} ms", std_latency);
    println!("    Min:    {:.3} ms", min_latency);
    println!("    Max:    {:.3} ms", max_latency);

    if peak_memory > 0 {
        println!(
            "  Peak Memory: {:.2} MB",
            peak_memory as f64 / 1_000_000.0
        );
    }

    // Throughput
    let batch_size = shape[0] as f64;
    let throughput = batch_size / (mean_latency / 1000.0);
    println!("  Throughput: {:.1} samples/sec", throughput);

    Ok(())
}

#[cfg(not(feature = "ffi"))]
fn run_without_ffi(
    model_path: &Path,
    config_path: &Path,
    device_str: &str,
    _dtype: &str,
    shape_str: &str,
) -> Result<()> {
    // Parse shape for the help message
    let shape: Vec<i64> = shape_str
        .split(',')
        .map(|s| s.trim().parse::<i64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap_or_default();

    println!("Inference Configuration:");
    println!("  Model: {:?}", model_path);
    println!("  Config: {:?}", config_path);
    println!();

    bail!(
        "Inference requires the FFI feature which is not enabled.\n\n\
         The FFI feature requires:\n\
         1. libtorch installed (set LIBTORCH environment variable)\n\
         2. C++ bridge built\n\n\
         To build with FFI support:\n\n\
         export LIBTORCH=/path/to/libtorch\n\
         cargo build --release --features ffi\n\n\
         To export a model for inference:\n\n\
         python scripts/export_aot.py \\\n\
             --config {:?} \\\n\
             --output {:?} \\\n\
             --shape {} \\\n\
             --device {}",
        config_path,
        model_path,
        shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
        device_str
    );
}
