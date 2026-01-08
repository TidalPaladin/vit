/**
 * @file vit_bridge.h
 * @brief C API for ViT inference using AOTInductor-compiled models.
 *
 * This header defines the C interface used by the Rust FFI bindings to
 * load and run inference on AOTInductor-packaged PyTorch models.
 *
 * Example usage from C:
 * @code
 *     VitModel* model = vit_model_load("model.pt2", "cuda:0");
 *     if (!model) {
 *         printf("Error: %s\n", vit_get_last_error());
 *         return 1;
 *     }
 *
 *     int64_t shape[] = {1, 3, 224, 224};
 *     float* data = ...; // Input data
 *     VitTensor* input = vit_tensor_create(data, shape, 4, "cuda:0");
 *
 *     VitInferenceResult* result = vit_model_infer(model, input);
 *     if (result) {
 *         printf("Latency: %.2f ms\n", vit_result_latency_ms(result));
 *         vit_result_free(result);
 *     }
 *
 *     vit_tensor_free(input);
 *     vit_model_free(model);
 * @endcode
 */

#ifndef VIT_BRIDGE_H
#define VIT_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque types */
typedef struct VitModel VitModel;
typedef struct VitTensor VitTensor;
typedef struct VitInferenceResult VitInferenceResult;

/**
 * @brief Load an AOTInductor model from a .pt2 file.
 *
 * @param pt2_path Path to the .pt2 model file.
 * @param device Device string (e.g., "cpu", "cuda:0").
 * @return Pointer to loaded model, or NULL on error. Use vit_get_last_error()
 *         for error details.
 */
VitModel* vit_model_load(const char* pt2_path, const char* device);

/**
 * @brief Free a loaded model.
 *
 * @param model Pointer to model to free. Safe to call with NULL.
 */
void vit_model_free(VitModel* model);

/**
 * @brief Get the device string for a loaded model.
 *
 * @param model Pointer to loaded model.
 * @return Device string (e.g., "cuda:0"). Valid until model is freed.
 */
const char* vit_model_device(const VitModel* model);

/**
 * @brief Create a tensor from float data.
 *
 * @param data Pointer to float data. Data is copied.
 * @param shape Pointer to shape array.
 * @param ndim Number of dimensions.
 * @param device Device string.
 * @return Pointer to created tensor, or NULL on error.
 */
VitTensor* vit_tensor_create(const float* data, const int64_t* shape,
                              size_t ndim, const char* device);

/**
 * @brief Create a tensor filled with random values.
 *
 * @param shape Pointer to shape array.
 * @param ndim Number of dimensions.
 * @param device Device string.
 * @return Pointer to created tensor, or NULL on error.
 */
VitTensor* vit_tensor_randn(const int64_t* shape, size_t ndim,
                             const char* device);

/**
 * @brief Free a tensor.
 *
 * @param tensor Pointer to tensor to free. Safe to call with NULL.
 */
void vit_tensor_free(VitTensor* tensor);

/**
 * @brief Get pointer to tensor data.
 *
 * The tensor is copied to CPU if necessary. The returned pointer is valid
 * until the tensor is freed.
 *
 * @param tensor Pointer to tensor.
 * @return Pointer to float data, or NULL on error.
 */
const float* vit_tensor_data(VitTensor* tensor);

/**
 * @brief Get tensor shape.
 *
 * @param tensor Pointer to tensor.
 * @return Pointer to shape array. Valid until tensor is freed.
 */
const int64_t* vit_tensor_shape(const VitTensor* tensor);

/**
 * @brief Get number of tensor dimensions.
 *
 * @param tensor Pointer to tensor.
 * @return Number of dimensions.
 */
size_t vit_tensor_ndim(const VitTensor* tensor);

/**
 * @brief Get total number of elements in tensor.
 *
 * @param tensor Pointer to tensor.
 * @return Number of elements.
 */
size_t vit_tensor_numel(const VitTensor* tensor);

/**
 * @brief Run inference on a model.
 *
 * @param model Pointer to loaded model.
 * @param input Input tensor.
 * @return Pointer to inference result, or NULL on error.
 */
VitInferenceResult* vit_model_infer(VitModel* model, VitTensor* input);

/**
 * @brief Get output tensor from inference result.
 *
 * The tensor is owned by the result and will be freed when the result is freed.
 *
 * @param result Pointer to inference result.
 * @return Pointer to output tensor.
 */
VitTensor* vit_result_tensor(VitInferenceResult* result);

/**
 * @brief Get inference latency in milliseconds.
 *
 * @param result Pointer to inference result.
 * @return Latency in milliseconds.
 */
double vit_result_latency_ms(const VitInferenceResult* result);

/**
 * @brief Get peak memory usage in bytes.
 *
 * Only accurate for CUDA devices. Returns 0 for CPU.
 *
 * @param result Pointer to inference result.
 * @return Memory usage in bytes.
 */
size_t vit_result_memory_bytes(const VitInferenceResult* result);

/**
 * @brief Free an inference result.
 *
 * Also frees the output tensor.
 *
 * @param result Pointer to result to free. Safe to call with NULL.
 */
void vit_result_free(VitInferenceResult* result);

/**
 * @brief Get the last error message.
 *
 * @return Error message string. Valid until next API call.
 */
const char* vit_get_last_error(void);

/**
 * @brief Check if CUDA is available.
 *
 * @return 1 if CUDA is available, 0 otherwise.
 */
int vit_cuda_available(void);

/**
 * @brief Get number of available CUDA devices.
 *
 * @return Number of CUDA devices, or 0 if CUDA is not available.
 */
int vit_cuda_device_count(void);

/**
 * @brief Synchronize CUDA device.
 *
 * Useful for accurate timing measurements.
 *
 * @param device_index CUDA device index.
 */
void vit_cuda_synchronize(int device_index);

#ifdef __cplusplus
}
#endif

#endif /* VIT_BRIDGE_H */
