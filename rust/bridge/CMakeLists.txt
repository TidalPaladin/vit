cmake_minimum_required(VERSION 3.18)
project(vit_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find libtorch
# Set LIBTORCH environment variable or CMAKE_PREFIX_PATH to your libtorch installation
if(DEFINED ENV{LIBTORCH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{LIBTORCH})
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Ensure we use the same C++ ABI as libtorch (new CXX11 ABI)
# This is required to avoid symbol mismatch errors with std::string
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# Check for CUDA or HIP/ROCm support
# PyTorch's HIP builds use TORCH_HIP_LIBRARIES, but the AOTInductor API is the same
if(USE_HIP OR DEFINED ENV{ROCM_PATH} OR DEFINED ROCM_PATH)
    # ROCm/HIP build - check for HIP libraries
    # Note: PyTorch ROCm builds may report TORCH_CUDA_LIBRARIES even though they use HIP
    message(STATUS "HIP/ROCm support enabled")
    add_compile_definitions(USE_HIP)
elseif(TORCH_CUDA_LIBRARIES)
    message(STATUS "CUDA support enabled")
    add_compile_definitions(USE_CUDA)
else()
    message(STATUS "GPU support disabled (CPU only)")
endif()

# Build the bridge library
add_library(vit_bridge SHARED
    src/vit_bridge.cpp
)

target_include_directories(vit_bridge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(vit_bridge
    PUBLIC
        ${TORCH_LIBRARIES}
)

# Set output directory and RPATH
set_target_properties(vit_bridge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    # Use $ORIGIN so the library looks for dependencies in the same directory
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install rules
install(TARGETS vit_bridge
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)

install(FILES include/vit_bridge.h
    DESTINATION include
)

# Print configuration summary
message(STATUS "=== vit_bridge Configuration ===")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
