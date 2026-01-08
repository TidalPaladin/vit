/**
 * @file vit_bridge.cpp
 * @brief Implementation of C API for ViT inference using AOTInductor.
 */

#include "vit_bridge.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

/* Thread-local error message */
static thread_local std::string g_last_error;

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

/* Internal structures */
struct VitModel {
    std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
    std::string device_str;
    torch::Device device;

    VitModel(std::unique_ptr<torch::inductor::AOTIModelContainerRunner> r,
             const std::string& dev_str, torch::Device dev)
        : runner(std::move(r)), device_str(dev_str), device(dev) {}
};

struct VitTensor {
    torch::Tensor tensor;
    std::vector<int64_t> shape_vec;
    mutable std::vector<float> cpu_data;  // Cached CPU copy
    mutable bool cpu_data_valid = false;

    explicit VitTensor(torch::Tensor t) : tensor(std::move(t)) {
        auto sizes = tensor.sizes();
        shape_vec.assign(sizes.begin(), sizes.end());
    }
};

struct VitInferenceResult {
    std::unique_ptr<VitTensor> output;
    double latency_ms;
    size_t memory_bytes;
};

/* Parse device string */
static torch::Device parse_device(const std::string& device_str) {
    if (device_str == "cpu") {
        return torch::kCPU;
    } else if (device_str.find("cuda") == 0) {
        if (device_str == "cuda") {
            return torch::Device(torch::kCUDA, 0);
        }
        // Parse "cuda:N"
        size_t colon = device_str.find(':');
        if (colon != std::string::npos) {
            int index = std::stoi(device_str.substr(colon + 1));
            return torch::Device(torch::kCUDA, index);
        }
        return torch::Device(torch::kCUDA, 0);
    }
    throw std::runtime_error("Invalid device: " + device_str);
}

/* C API Implementation */

extern "C" {

VitModel* vit_model_load(const char* pt2_path, const char* device_str) {
    try {
        std::string path(pt2_path);
        std::string dev_str(device_str);
        torch::Device device = parse_device(dev_str);

        std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;

        if (device.is_cuda()) {
#ifdef USE_CUDA
            runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(path);
#else
            set_error("CUDA support not compiled in");
            return nullptr;
#endif
        } else {
            runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(path);
        }

        return new VitModel(std::move(runner), dev_str, device);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

void vit_model_free(VitModel* model) {
    delete model;
}

const char* vit_model_device(const VitModel* model) {
    if (!model) return "";
    return model->device_str.c_str();
}

VitTensor* vit_tensor_create(const float* data, const int64_t* shape,
                              size_t ndim, const char* device_str) {
    try {
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto options = torch::TensorOptions().dtype(torch::kFloat32);

        // Create tensor on CPU first
        torch::Tensor tensor = torch::from_blob(
            const_cast<float*>(data),
            shape_vec,
            options
        ).clone();  // Clone to own the data

        // Move to device if needed
        torch::Device device = parse_device(device_str);
        if (!device.is_cpu()) {
            tensor = tensor.to(device);
        }

        return new VitTensor(std::move(tensor));
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

VitTensor* vit_tensor_randn(const int64_t* shape, size_t ndim,
                             const char* device_str) {
    try {
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        torch::Device device = parse_device(device_str);
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device);

        torch::Tensor tensor = torch::randn(shape_vec, options);
        return new VitTensor(std::move(tensor));
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

void vit_tensor_free(VitTensor* tensor) {
    delete tensor;
}

const float* vit_tensor_data(VitTensor* tensor) {
    if (!tensor) return nullptr;
    try {
        // Copy to CPU if needed
        if (!tensor->tensor.is_cpu()) {
            if (!tensor->cpu_data_valid) {
                torch::Tensor cpu_tensor = tensor->tensor.to(torch::kCPU).contiguous();
                tensor->cpu_data.resize(cpu_tensor.numel());
                std::memcpy(tensor->cpu_data.data(), cpu_tensor.data_ptr<float>(),
                           cpu_tensor.numel() * sizeof(float));
                tensor->cpu_data_valid = true;
            }
            return tensor->cpu_data.data();
        }
        return tensor->tensor.contiguous().data_ptr<float>();
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

const int64_t* vit_tensor_shape(const VitTensor* tensor) {
    if (!tensor) return nullptr;
    return tensor->shape_vec.data();
}

size_t vit_tensor_ndim(const VitTensor* tensor) {
    if (!tensor) return 0;
    return tensor->shape_vec.size();
}

size_t vit_tensor_numel(const VitTensor* tensor) {
    if (!tensor) return 0;
    return tensor->tensor.numel();
}

VitInferenceResult* vit_model_infer(VitModel* model, VitTensor* input) {
    if (!model || !input) {
        set_error("Null model or input");
        return nullptr;
    }

    try {
        // Move input to model's device if needed
        torch::Tensor input_tensor = input->tensor;
        if (input_tensor.device() != model->device) {
            input_tensor = input_tensor.to(model->device);
        }

        // Prepare inputs
        std::vector<torch::Tensor> inputs = {input_tensor};

        // Get memory before (CUDA only)
        size_t memory_before = 0;
#ifdef USE_CUDA
        if (model->device.is_cuda()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
            auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(model->device.index());
            memory_before = stats.allocated_bytes[0].current;
        }
#endif

        // Synchronize before timing
#ifdef USE_CUDA
        if (model->device.is_cuda()) {
            c10::cuda::device_synchronize();
        }
#endif

        // Run inference with timing
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<torch::Tensor> outputs = model->runner->run(inputs);

        // Synchronize after inference
#ifdef USE_CUDA
        if (model->device.is_cuda()) {
            c10::cuda::device_synchronize();
        }
#endif

        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Get memory after (CUDA only)
        size_t memory_after = 0;
#ifdef USE_CUDA
        if (model->device.is_cuda()) {
            auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(model->device.index());
            memory_after = stats.allocated_bytes[0].peak;
        }
#endif

        if (outputs.empty()) {
            set_error("Model returned no outputs");
            return nullptr;
        }

        auto result = new VitInferenceResult();
        result->output = std::make_unique<VitTensor>(outputs[0]);
        result->latency_ms = latency_ms;
        result->memory_bytes = memory_after > memory_before ? memory_after - memory_before : 0;

        return result;
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

VitTensor* vit_result_tensor(VitInferenceResult* result) {
    if (!result) return nullptr;
    return result->output.get();
}

double vit_result_latency_ms(const VitInferenceResult* result) {
    if (!result) return 0.0;
    return result->latency_ms;
}

size_t vit_result_memory_bytes(const VitInferenceResult* result) {
    if (!result) return 0;
    return result->memory_bytes;
}

void vit_result_free(VitInferenceResult* result) {
    delete result;
}

const char* vit_get_last_error(void) {
    return g_last_error.c_str();
}

int vit_cuda_available(void) {
#ifdef USE_CUDA
    return torch::cuda::is_available() ? 1 : 0;
#else
    return 0;
#endif
}

int vit_cuda_device_count(void) {
#ifdef USE_CUDA
    return torch::cuda::device_count();
#else
    return 0;
#endif
}

void vit_cuda_synchronize(int device_index) {
#ifdef USE_CUDA
    c10::cuda::device_synchronize();
#endif
    (void)device_index;
}

} // extern "C"
