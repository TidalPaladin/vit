//! Integration tests for vit-ffi.
//!
//! These tests exercise the FFI bridge with actual libtorch operations.
//! Tests skip gracefully if the FFI bridge is not available (LIBTORCH not set).
//!
//! Run with: cargo test --package vit-ffi --test integration

use vit_ffi::{cuda_available, cuda_device_count, is_available, Device, Model, Tensor};

/// Helper to skip tests when FFI bridge is not available.
fn skip_if_no_ffi() -> bool {
    if !is_available() {
        eprintln!("Skipping integration test: FFI bridge not available (LIBTORCH not set)");
        true
    } else {
        false
    }
}

/// Get the best available device (GPU if available, otherwise CPU).
fn best_device() -> Device {
    if cuda_available() && cuda_device_count() > 0 {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

// ==================== Tensor Integration Tests ====================

#[test]
fn test_tensor_cpu_creation_various_shapes() {
    if skip_if_no_ffi() {
        return;
    }

    let test_cases = vec![
        vec![1i64],           // Scalar-ish
        vec![10],             // 1D
        vec![3, 4],           // 2D
        vec![2, 3, 4],        // 3D
        vec![1, 3, 224, 224], // Typical image batch
    ];

    for shape in test_cases {
        let tensor = Tensor::randn(&shape, Device::Cpu);
        assert!(tensor.is_ok(), "Failed to create tensor with shape {:?}", shape);
        let t = tensor.unwrap();
        assert_eq!(t.shape(), shape, "Shape mismatch for {:?}", shape);
        let expected_numel: i64 = shape.iter().product();
        assert_eq!(t.numel(), expected_numel as usize);
    }
}

#[test]
fn test_tensor_data_integrity() {
    if skip_if_no_ffi() {
        return;
    }

    // Create tensor with known data
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let shape = vec![2i64, 3, 4];
    let tensor = Tensor::from_slice(&data, &shape, Device::Cpu).unwrap();

    // Extract and verify
    let extracted = tensor.to_vec().unwrap();
    assert_eq!(extracted.len(), data.len());
    for (i, (a, b)) in data.iter().zip(extracted.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "Mismatch at index {}: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_tensor_on_best_device() {
    if skip_if_no_ffi() {
        return;
    }

    let device = best_device();
    let shape = vec![2i64, 3, 224, 224];
    let tensor = Tensor::randn(&shape, device);

    assert!(
        tensor.is_ok(),
        "Failed to create tensor on {:?}: {:?}",
        device,
        tensor.err()
    );
}

#[test]
fn test_tensor_gpu_to_cpu_transfer() {
    if skip_if_no_ffi() {
        return;
    }

    let device = best_device();
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let shape = vec![10i64, 10];

    let tensor = Tensor::from_slice(&data, &shape, device).unwrap();

    // to_vec() should work regardless of device (copies to CPU internally)
    let extracted = tensor.to_vec();
    assert!(
        extracted.is_ok(),
        "Failed to extract data from {:?} tensor",
        device
    );

    let extracted = extracted.unwrap();
    assert_eq!(extracted.len(), 100);

    // Verify data integrity
    for (i, (a, b)) in data.iter().zip(extracted.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "Data mismatch at index {}: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_multiple_tensor_allocation() {
    if skip_if_no_ffi() {
        return;
    }

    // Allocate multiple tensors to test memory management
    let mut tensors = Vec::new();
    for i in 0..10 {
        let shape = vec![100i64, 100];
        let tensor = Tensor::randn(&shape, Device::Cpu);
        assert!(tensor.is_ok(), "Failed to allocate tensor {}", i);
        tensors.push(tensor.unwrap());
    }

    // Verify all tensors are valid
    for (i, tensor) in tensors.iter().enumerate() {
        assert_eq!(tensor.numel(), 10000, "Tensor {} has wrong numel", i);
    }

    // Drop tensors (tests cleanup)
    drop(tensors);
}

// ==================== Model Loading Tests ====================

#[test]
fn test_model_load_invalid_path() {
    if skip_if_no_ffi() {
        return;
    }

    let result = Model::load("/nonexistent/path/to/model.so", Device::Cpu);
    assert!(result.is_err(), "Loading nonexistent model should fail");

    match result {
        Err(err) => {
            let msg = format!("{}", err);
            assert!(!msg.is_empty(), "Error message should not be empty");
        }
        Ok(_) => panic!("Expected error but got Ok"),
    }
}

#[test]
fn test_model_load_invalid_file() {
    if skip_if_no_ffi() {
        return;
    }

    // Try to load a non-.so file (like Cargo.toml)
    let result = Model::load("/home/tidal/Documents/vit/rust/vit-ffi/Cargo.toml", Device::Cpu);
    assert!(result.is_err(), "Loading non-.so file should fail");
}

// ==================== CUDA Utility Tests ====================

#[test]
fn test_cuda_available_consistency() {
    if skip_if_no_ffi() {
        return;
    }

    // Call multiple times, should be consistent
    let first = cuda_available();
    let second = cuda_available();
    assert_eq!(first, second, "cuda_available() should be consistent");
}

#[test]
fn test_cuda_device_count_consistency() {
    if skip_if_no_ffi() {
        return;
    }

    let first = cuda_device_count();
    let second = cuda_device_count();
    assert_eq!(first, second, "cuda_device_count() should be consistent");

    // If CUDA is available, device count should be > 0
    if cuda_available() {
        assert!(first > 0, "CUDA available but device count is 0");
    }
}

#[test]
fn test_cuda_available_matches_device_count() {
    if skip_if_no_ffi() {
        return;
    }

    let available = cuda_available();
    let count = cuda_device_count();

    if available {
        assert!(count > 0, "CUDA available but no devices found");
    }
    // Note: count could be 0 even if CUDA libraries are present but no GPU
}

// ==================== Device Tests with FFI ====================

#[test]
fn test_device_cpu_tensor_operations() {
    if skip_if_no_ffi() {
        return;
    }

    let device = Device::Cpu;

    // Create, inspect, and extract
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_slice(&data, &[2, 2], device).unwrap();

    assert_eq!(tensor.shape(), vec![2, 2]);
    assert_eq!(tensor.ndim(), 2);
    assert_eq!(tensor.numel(), 4);

    let extracted = tensor.to_vec().unwrap();
    assert_eq!(extracted, data);
}

#[test]
fn test_device_cuda_tensor_if_available() {
    if skip_if_no_ffi() {
        return;
    }

    if !cuda_available() || cuda_device_count() == 0 {
        eprintln!("Skipping CUDA tensor test: no CUDA devices available");
        return;
    }

    let device = Device::Cuda(0);
    let shape = vec![4i64, 4];
    let tensor = Tensor::randn(&shape, device);

    assert!(tensor.is_ok(), "Failed to create CUDA tensor: {:?}", tensor.err());
    let t = tensor.unwrap();
    assert_eq!(t.shape(), shape);

    // Should be able to extract data (copies to CPU)
    let data = t.to_vec();
    assert!(data.is_ok(), "Failed to extract CUDA tensor data");
    assert_eq!(data.unwrap().len(), 16);
}

// ==================== Stress Tests ====================

#[test]
fn test_rapid_tensor_creation_deletion() {
    if skip_if_no_ffi() {
        return;
    }

    // Create and immediately drop many tensors
    for _ in 0..100 {
        let tensor = Tensor::randn(&[10, 10], Device::Cpu);
        assert!(tensor.is_ok());
        // Tensor dropped here
    }
}

#[test]
fn test_large_tensor_allocation() {
    if skip_if_no_ffi() {
        return;
    }

    // Allocate a moderately large tensor (~16MB for f32)
    let shape = vec![1024i64, 1024, 4];
    let tensor = Tensor::randn(&shape, Device::Cpu);

    assert!(tensor.is_ok(), "Failed to allocate large tensor");
    let t = tensor.unwrap();
    assert_eq!(t.numel(), 1024 * 1024 * 4);
}
