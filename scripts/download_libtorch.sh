#!/usr/bin/env bash
#
# Download and install libtorch with CUDA support.
#
# Usage:
#   ./scripts/download_libtorch.sh [OPTIONS]
#
# Options:
#   --cuda VERSION    CUDA version (11.8, 12.1, 12.4, cpu) [default: 12.4]
#   --output DIR      Output directory [default: ./libtorch]
#   --cxx11-abi       Use CXX11 ABI (Pre-cxx11 ABI is default)
#   --help            Show this help message
#
# Examples:
#   ./scripts/download_libtorch.sh --cuda 12.4
#   ./scripts/download_libtorch.sh --cuda cpu --output /opt/libtorch
#
# After installation, set LIBTORCH environment variable:
#   export LIBTORCH=/path/to/libtorch
#

set -euo pipefail

# Default values
CUDA_VERSION="12.8"
OUTPUT_DIR="./libtorch"
CXX11_ABI=false
PYTORCH_VERSION="2.9.1"  # Update as needed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    head -25 "$0" | tail -22 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cxx11-abi)
            CXX11_ABI=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)
        PLATFORM="linux"
        ;;
    Darwin*)
        PLATFORM="macos"
        if [[ "$CUDA_VERSION" != "cpu" ]]; then
            log_warn "CUDA is not supported on macOS, forcing CPU version"
            CUDA_VERSION="cpu"
        fi
        ;;
    *)
        log_error "Unsupported OS: $OS"
        exit 1
        ;;
esac

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        if [[ "$CUDA_VERSION" != "cpu" ]]; then
            log_warn "CUDA builds not available for ARM64, forcing CPU version"
            CUDA_VERSION="cpu"
        fi
        ;;
    *)
        log_error "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Build download URL
# PyTorch download URLs follow this pattern:
# https://download.pytorch.org/libtorch/{cu|cpu}/{variant}/libtorch-{abi}-{version}%2B{cuda}.zip

if [[ "$CXX11_ABI" == true ]]; then
    ABI_VARIANT="cxx11-abi-shared-with-deps"
else
    ABI_VARIANT="shared-with-deps"
fi

case "$CUDA_VERSION" in
    cpu)
        CUDA_TAG="cpu"
        DOWNLOAD_PATH="cpu"
        ;;
    11.8)
        CUDA_TAG="cu118"
        DOWNLOAD_PATH="cu118"
        ;;
    12.1)
        CUDA_TAG="cu121"
        DOWNLOAD_PATH="cu121"
        ;;
    12.4)
        CUDA_TAG="cu124"
        DOWNLOAD_PATH="cu124"
        ;;
    12.8)
        CUDA_TAG="cu128"
        DOWNLOAD_PATH="cu128"
        ;;
    13.0)
        CUDA_TAG="cu130"
        DOWNLOAD_PATH="cu130"
        ;;
    *)
        log_error "Unsupported CUDA version: $CUDA_VERSION"
        log_error "Supported versions: cpu, 11.8, 12.1, 12.4, 12.8, 13.0"
        exit 1
        ;;
esac

if [[ "$PLATFORM" == "macos" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        FILENAME="libtorch-macos-arm64-${PYTORCH_VERSION}.zip"
    else
        FILENAME="libtorch-macos-x86_64-${PYTORCH_VERSION}.zip"
    fi
    DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/${FILENAME}"
else
    FILENAME="libtorch-${ABI_VARIANT}-${PYTORCH_VERSION}%2B${CUDA_TAG}.zip"
    DOWNLOAD_URL="https://download.pytorch.org/libtorch/${DOWNLOAD_PATH}/${FILENAME}"
fi

log_info "Configuration:"
log_info "  Platform:      $PLATFORM ($ARCH)"
log_info "  CUDA version:  $CUDA_VERSION"
log_info "  CXX11 ABI:     $CXX11_ABI"
log_info "  PyTorch:       $PYTORCH_VERSION"
log_info "  Output:        $OUTPUT_DIR"
echo

# Check for required tools
for cmd in curl unzip; do
    if ! command -v "$cmd" &> /dev/null; then
        log_error "$cmd is required but not installed"
        exit 1
    fi
done

# Create output directory
mkdir -p "$(dirname "$OUTPUT_DIR")"

# Download
TEMP_ZIP=$(mktemp /tmp/libtorch-XXXXXX.zip)
trap "rm -f $TEMP_ZIP" EXIT

log_info "Downloading libtorch from:"
log_info "  $DOWNLOAD_URL"
echo

if ! curl -L --progress-bar -o "$TEMP_ZIP" "$DOWNLOAD_URL"; then
    log_error "Download failed"
    log_error "URL: $DOWNLOAD_URL"
    exit 1
fi

# Check file size (should be > 100MB for a valid download)
FILE_SIZE=$(stat -f%z "$TEMP_ZIP" 2>/dev/null || stat -c%s "$TEMP_ZIP" 2>/dev/null)
if [[ "$FILE_SIZE" -lt 100000000 ]]; then
    log_error "Downloaded file is too small (${FILE_SIZE} bytes)"
    log_error "This might indicate a download error or incorrect URL"
    exit 1
fi

# Extract
log_info "Extracting to $OUTPUT_DIR..."

# Remove existing installation
if [[ -d "$OUTPUT_DIR" ]]; then
    log_warn "Removing existing installation at $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# Extract to parent directory (zip contains 'libtorch' folder)
PARENT_DIR="$(dirname "$OUTPUT_DIR")"
unzip -q "$TEMP_ZIP" -d "$PARENT_DIR"

# Rename if needed
EXTRACTED_DIR="$PARENT_DIR/libtorch"
if [[ "$EXTRACTED_DIR" != "$OUTPUT_DIR" ]] && [[ -d "$EXTRACTED_DIR" ]]; then
    mv "$EXTRACTED_DIR" "$OUTPUT_DIR"
fi

# Verify installation
if [[ ! -f "$OUTPUT_DIR/lib/libtorch.so" ]] && [[ ! -f "$OUTPUT_DIR/lib/libtorch.dylib" ]]; then
    log_error "Installation verification failed"
    log_error "Expected library not found in $OUTPUT_DIR/lib/"
    exit 1
fi

log_info "Installation complete!"
echo
log_info "Add the following to your shell profile (.bashrc, .zshrc, etc.):"
echo
echo "  export LIBTORCH=\"$(cd "$OUTPUT_DIR" && pwd)\""
if [[ "$CXX11_ABI" == true ]]; then
    echo "  export LIBTORCH_CXX11_ABI=1"
fi
echo
log_info "Then build the Rust CLI with FFI support:"
echo
echo "  make rust-ffi"
echo
