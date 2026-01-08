//! FFI bindings to C++ AOTInductor runtime for ViT inference.
//!
//! This crate provides safe Rust wrappers around the C++ bridge library
//! that loads and runs AOTInductor-compiled PyTorch models.
//!
//! # Building
//!
//! This crate requires:
//! - libtorch (matching your PyTorch version)
//! - CMake 3.18+
//! - C++17 compiler
//! - CUDA toolkit (optional, for GPU support)
//!
//! Set the `LIBTORCH` environment variable to your libtorch installation:
//!
//! ```bash
//! export LIBTORCH=/path/to/libtorch
//! cargo build
//! ```
//!
//! # Example
//!
//! ```ignore
//! use vit_ffi::{Model, Tensor, Device};
//!
//! let model = Model::load("model.pt2", Device::Cuda(0))?;
//! let input = Tensor::randn(&[1, 3, 224, 224], Device::Cuda(0))?;
//! let result = model.infer(&input)?;
//!
//! println!("Latency: {:.2} ms", result.latency_ms());
//! println!("Output shape: {:?}", result.output().shape());
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::path::Path;

/// Error types for vit-ffi operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// FFI not available (bridge not built).
    #[error("FFI not available: {0}")]
    NotAvailable(String),

    /// Failed to load model.
    #[error("failed to load model: {0}")]
    LoadModel(String),

    /// Inference failed.
    #[error("inference failed: {0}")]
    Inference(String),

    /// Invalid device specification.
    #[error("invalid device: {0}")]
    InvalidDevice(String),

    /// Null pointer error.
    #[error("null pointer returned: {0}")]
    NullPointer(String),

    /// Invalid path.
    #[error("invalid path: {0}")]
    InvalidPath(String),
}

/// Result type for vit-ffi operations.
pub type Result<T> = std::result::Result<T, Error>;

// FFI declarations - these match the C API in vit_bridge.h
mod ffi {
    use std::os::raw::c_int;

    #[repr(C)]
    pub struct VitModel {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct VitTensor {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct VitInferenceResult {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        // Model lifecycle
        pub fn vit_model_load(pt2_path: *const i8, device: *const i8) -> *mut VitModel;
        pub fn vit_model_free(model: *mut VitModel);
        pub fn vit_model_device(model: *const VitModel) -> *const i8;

        // Tensor operations
        pub fn vit_tensor_create(
            data: *const f32,
            shape: *const i64,
            ndim: usize,
            device: *const i8,
        ) -> *mut VitTensor;
        pub fn vit_tensor_randn(
            shape: *const i64,
            ndim: usize,
            device: *const i8,
        ) -> *mut VitTensor;
        pub fn vit_tensor_free(tensor: *mut VitTensor);
        pub fn vit_tensor_data(tensor: *mut VitTensor) -> *const f32;
        pub fn vit_tensor_shape(tensor: *const VitTensor) -> *const i64;
        pub fn vit_tensor_ndim(tensor: *const VitTensor) -> usize;
        pub fn vit_tensor_numel(tensor: *const VitTensor) -> usize;

        // Inference
        pub fn vit_model_infer(
            model: *mut VitModel,
            input: *mut VitTensor,
        ) -> *mut VitInferenceResult;
        pub fn vit_result_tensor(result: *mut VitInferenceResult) -> *mut VitTensor;
        pub fn vit_result_latency_ms(result: *const VitInferenceResult) -> f64;
        pub fn vit_result_memory_bytes(result: *const VitInferenceResult) -> usize;
        pub fn vit_result_free(result: *mut VitInferenceResult);

        // Error handling
        pub fn vit_get_last_error() -> *const i8;

        // CUDA utilities
        pub fn vit_cuda_available() -> c_int;
        pub fn vit_cuda_device_count() -> c_int;
        pub fn vit_cuda_synchronize(device_index: c_int);
    }
}

/// Check if the FFI bridge is available.
///
/// Returns true if the C++ bridge was built and linked successfully.
pub fn is_available() -> bool {
    // Try to call a simple FFI function
    // If the library isn't linked, this will return a sensible default
    unsafe { ffi::vit_cuda_device_count() >= 0 }
}

/// Get the last error message from the C++ bridge.
fn get_last_error() -> String {
    unsafe {
        let ptr = ffi::vit_get_last_error();
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Device specification for tensor and model placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU device.
    Cpu,
    /// CUDA device with index.
    Cuda(usize),
}

impl Device {
    /// Parse a device string like "cpu" or "cuda:0".
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_lowercase();
        if s == "cpu" {
            Ok(Device::Cpu)
        } else if s.starts_with("cuda") {
            let idx = if s == "cuda" {
                0
            } else if let Some(rest) = s.strip_prefix("cuda:") {
                rest.parse::<usize>()
                    .map_err(|_| Error::InvalidDevice(s.clone()))?
            } else {
                return Err(Error::InvalidDevice(s));
            };
            Ok(Device::Cuda(idx))
        } else {
            Err(Error::InvalidDevice(s))
        }
    }

    /// Convert to a C string for FFI.
    fn to_c_string(&self) -> CString {
        let s = match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(idx) => format!("cuda:{}", idx),
        };
        CString::new(s).unwrap()
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{}", idx),
        }
    }
}

/// Check if CUDA is available.
pub fn cuda_available() -> bool {
    unsafe { ffi::vit_cuda_available() != 0 }
}

/// Get the number of available CUDA devices.
pub fn cuda_device_count() -> usize {
    unsafe { ffi::vit_cuda_device_count() as usize }
}

/// Synchronize a CUDA device.
pub fn cuda_synchronize(device_index: usize) {
    unsafe {
        ffi::vit_cuda_synchronize(device_index as c_int);
    }
}

/// A tensor on a specific device.
pub struct Tensor {
    ptr: *mut ffi::VitTensor,
    owned: bool,
}

// Tensor is Send because the underlying C++ tensor is thread-safe
unsafe impl Send for Tensor {}

impl Tensor {
    /// Create a tensor from float data.
    ///
    /// The data is copied to the specified device.
    pub fn from_slice(data: &[f32], shape: &[i64], device: Device) -> Result<Self> {
        let device_str = device.to_c_string();
        let ptr = unsafe {
            ffi::vit_tensor_create(
                data.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                device_str.as_ptr(),
            )
        };
        if ptr.is_null() {
            return Err(Error::NullPointer(get_last_error()));
        }
        Ok(Tensor { ptr, owned: true })
    }

    /// Create a tensor filled with random values.
    pub fn randn(shape: &[i64], device: Device) -> Result<Self> {
        let device_str = device.to_c_string();
        let ptr = unsafe { ffi::vit_tensor_randn(shape.as_ptr(), shape.len(), device_str.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::NullPointer(get_last_error()));
        }
        Ok(Tensor { ptr, owned: true })
    }

    /// Wrap a raw pointer (non-owning).
    unsafe fn from_ptr(ptr: *mut ffi::VitTensor) -> Self {
        Tensor { ptr, owned: false }
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<i64> {
        unsafe {
            let ndim = ffi::vit_tensor_ndim(self.ptr);
            let shape_ptr = ffi::vit_tensor_shape(self.ptr);
            if shape_ptr.is_null() || ndim == 0 {
                return vec![];
            }
            std::slice::from_raw_parts(shape_ptr, ndim).to_vec()
        }
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { ffi::vit_tensor_ndim(self.ptr) }
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        unsafe { ffi::vit_tensor_numel(self.ptr) }
    }

    /// Get the tensor data as a slice.
    ///
    /// If the tensor is on a GPU, it will be copied to CPU first.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        unsafe {
            let data_ptr = ffi::vit_tensor_data(self.ptr as *mut _);
            if data_ptr.is_null() {
                return Err(Error::NullPointer(get_last_error()));
            }
            let numel = self.numel();
            Ok(std::slice::from_raw_parts(data_ptr, numel).to_vec())
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe {
                ffi::vit_tensor_free(self.ptr);
            }
        }
    }
}

/// Result of an inference operation.
pub struct InferenceResult {
    ptr: *mut ffi::VitInferenceResult,
}

// InferenceResult is Send because the underlying C++ result is thread-safe
unsafe impl Send for InferenceResult {}

impl InferenceResult {
    /// Get the output tensor.
    ///
    /// The tensor is owned by this result and will be freed when the result is dropped.
    pub fn output(&self) -> Tensor {
        unsafe {
            let tensor_ptr = ffi::vit_result_tensor(self.ptr);
            Tensor::from_ptr(tensor_ptr)
        }
    }

    /// Get the inference latency in milliseconds.
    pub fn latency_ms(&self) -> f64 {
        unsafe { ffi::vit_result_latency_ms(self.ptr) }
    }

    /// Get the peak memory usage in bytes.
    ///
    /// Only accurate for CUDA devices. Returns 0 for CPU.
    pub fn memory_bytes(&self) -> usize {
        unsafe { ffi::vit_result_memory_bytes(self.ptr) }
    }
}

impl Drop for InferenceResult {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::vit_result_free(self.ptr);
            }
        }
    }
}

/// A loaded AOTInductor model.
pub struct Model {
    ptr: *mut ffi::VitModel,
    device: Device,
}

// Model is Send because the underlying C++ model is thread-safe for inference
unsafe impl Send for Model {}

impl Model {
    /// Load a model from a .pt2 file.
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| Error::InvalidPath(path.display().to_string()))?;
        let path_c = CString::new(path_str)
            .map_err(|_| Error::InvalidPath(path.display().to_string()))?;
        let device_str = device.to_c_string();

        let ptr = unsafe { ffi::vit_model_load(path_c.as_ptr(), device_str.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::LoadModel(get_last_error()));
        }

        Ok(Model { ptr, device })
    }

    /// Get the device this model is on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run inference on the model.
    pub fn infer(&self, input: &Tensor) -> Result<InferenceResult> {
        let result_ptr =
            unsafe { ffi::vit_model_infer(self.ptr, input.ptr as *mut ffi::VitTensor) };
        if result_ptr.is_null() {
            return Err(Error::Inference(get_last_error()));
        }
        Ok(InferenceResult { ptr: result_ptr })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::vit_model_free(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Device Parsing Tests ====================

    #[test]
    fn test_parse_device_cpu() {
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("CPU").unwrap(), Device::Cpu);
    }

    #[test]
    fn test_parse_device_cuda() {
        assert_eq!(Device::parse("cuda").unwrap(), Device::Cuda(0));
        assert_eq!(Device::parse("cuda:0").unwrap(), Device::Cuda(0));
        assert_eq!(Device::parse("cuda:1").unwrap(), Device::Cuda(1));
        assert_eq!(Device::parse("CUDA:2").unwrap(), Device::Cuda(2));
    }

    #[test]
    fn test_parse_device_invalid() {
        assert!(Device::parse("gpu").is_err());
        assert!(Device::parse("cuda:abc").is_err());
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
    }

    #[test]
    fn test_device_parse_with_whitespace() {
        assert_eq!(Device::parse("  cpu  ").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("\tcuda:0\n").unwrap(), Device::Cuda(0));
    }

    #[test]
    fn test_device_parse_cuda_high_index() {
        // High indices should parse (validation happens at use time)
        assert_eq!(Device::parse("cuda:99").unwrap(), Device::Cuda(99));
        assert_eq!(Device::parse("cuda:255").unwrap(), Device::Cuda(255));
    }

    #[test]
    fn test_device_parse_invalid_formats() {
        // Empty string
        assert!(Device::parse("").is_err());
        // Unknown device type
        assert!(Device::parse("tpu").is_err());
        assert!(Device::parse("mps").is_err());
        // Malformed cuda
        assert!(Device::parse("cuda:").is_err());
        assert!(Device::parse("cuda::0").is_err());
        assert!(Device::parse("cuda:-1").is_err());
        assert!(Device::parse("cudaa").is_err());
    }

    #[test]
    fn test_device_display_roundtrip() {
        // Parse -> Display -> Parse should be idempotent
        let devices = vec![Device::Cpu, Device::Cuda(0), Device::Cuda(7)];
        for device in devices {
            let s = device.to_string();
            let parsed = Device::parse(&s).unwrap();
            assert_eq!(device, parsed);
        }
    }

    #[test]
    fn test_device_equality() {
        assert_eq!(Device::Cpu, Device::Cpu);
        assert_eq!(Device::Cuda(0), Device::Cuda(0));
        assert_ne!(Device::Cpu, Device::Cuda(0));
        assert_ne!(Device::Cuda(0), Device::Cuda(1));
    }

    #[test]
    fn test_device_clone() {
        let d1 = Device::Cuda(5);
        let d2 = d1;
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_device_debug() {
        let d = Device::Cuda(3);
        let debug_str = format!("{:?}", d);
        assert!(debug_str.contains("Cuda"));
        assert!(debug_str.contains("3"));
    }

    // ==================== Error Type Tests ====================

    #[test]
    fn test_error_display_not_available() {
        let err = Error::NotAvailable("bridge not linked".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("not available"));
        assert!(msg.contains("bridge not linked"));
    }

    #[test]
    fn test_error_display_load_model() {
        let err = Error::LoadModel("file not found".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("load model"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn test_error_display_inference() {
        let err = Error::Inference("out of memory".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("inference"));
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn test_error_display_invalid_device() {
        let err = Error::InvalidDevice("xyz".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("invalid device"));
        assert!(msg.contains("xyz"));
    }

    #[test]
    fn test_error_display_null_pointer() {
        let err = Error::NullPointer("tensor creation failed".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("null pointer"));
        assert!(msg.contains("tensor creation failed"));
    }

    #[test]
    fn test_error_display_invalid_path() {
        let err = Error::InvalidPath("/bad/path".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("invalid path"));
        assert!(msg.contains("/bad/path"));
    }

    #[test]
    fn test_error_debug_impl() {
        let err = Error::LoadModel("test error".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("LoadModel"));
        assert!(debug_str.contains("test error"));
    }

    // ==================== Conditional FFI Tests ====================
    // These tests skip gracefully when FFI is not available

    fn require_ffi() -> bool {
        if !is_available() {
            eprintln!("Skipping: FFI bridge not available (LIBTORCH not set)");
            false
        } else {
            true
        }
    }

    #[test]
    fn test_cuda_available_returns_bool() {
        if !require_ffi() {
            return;
        }
        // Should not panic, returns a boolean
        let _ = cuda_available();
    }

    #[test]
    fn test_cuda_device_count_non_negative() {
        if !require_ffi() {
            return;
        }
        let count = cuda_device_count();
        // Count is always >= 0 (it's usize)
        assert!(count < 1000); // Sanity check
    }

    #[test]
    fn test_tensor_create_cpu() {
        if !require_ffi() {
            return;
        }
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2i64, 2];
        let tensor = Tensor::from_slice(&data, &shape, Device::Cpu);
        assert!(tensor.is_ok(), "Failed to create tensor: {:?}", tensor.err());
        let t = tensor.unwrap();
        assert_eq!(t.shape(), vec![2, 2]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_tensor_randn_cpu() {
        if !require_ffi() {
            return;
        }
        let shape = vec![2i64, 3, 4];
        let tensor = Tensor::randn(&shape, Device::Cpu);
        assert!(tensor.is_ok(), "Failed to create randn tensor: {:?}", tensor.err());
        let t = tensor.unwrap();
        assert_eq!(t.shape(), shape);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_tensor_to_vec_roundtrip() {
        if !require_ffi() {
            return;
        }
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2i64, 3];
        let tensor = Tensor::from_slice(&data, &shape, Device::Cpu).unwrap();
        let extracted = tensor.to_vec().unwrap();
        assert_eq!(extracted, data);
    }

    #[test]
    fn test_tensor_scalar() {
        if !require_ffi() {
            return;
        }
        let data = vec![42.0f32];
        let shape = vec![1i64];
        let tensor = Tensor::from_slice(&data, &shape, Device::Cpu);
        assert!(tensor.is_ok());
        let t = tensor.unwrap();
        assert_eq!(t.numel(), 1);
    }

    #[test]
    fn test_tensor_1d() {
        if !require_ffi() {
            return;
        }
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let shape = vec![100i64];
        let tensor = Tensor::from_slice(&data, &shape, Device::Cpu).unwrap();
        assert_eq!(tensor.shape(), vec![100]);
        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.numel(), 100);
        let extracted = tensor.to_vec().unwrap();
        assert_eq!(extracted, data);
    }

    #[test]
    fn test_tensor_on_available_device() {
        if !require_ffi() {
            return;
        }
        // Use GPU if available, otherwise CPU
        let device = if cuda_available() && cuda_device_count() > 0 {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let shape = vec![4i64, 4];
        let tensor = Tensor::randn(&shape, device);
        assert!(tensor.is_ok(), "Failed to create tensor on {:?}: {:?}", device, tensor.err());
    }

    #[test]
    fn test_model_load_nonexistent_path() {
        if !require_ffi() {
            return;
        }
        let result = Model::load("/nonexistent/path/to/model.so", Device::Cpu);
        assert!(result.is_err());
        match result {
            Err(err) => {
                let msg = format!("{}", err);
                assert!(!msg.is_empty(), "Error message should not be empty");
            }
            Ok(_) => panic!("Expected error but got Ok"),
        }
    }
}
