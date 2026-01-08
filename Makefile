.PHONY: clean clean-env check quality style tag-version test env upload upload-test
.PHONY: rust rust-release rust-ffi rust-ffi-rocm rust-install rust-test rust-test-ffi rust-clean rust-check

PROJECT=vit
QUALITY_DIRS=$(PROJECT) tests benchmark scripts
CLEAN_DIRS=$(PROJECT) tests benchmark
PYTHON=uv run python
CARGO=cargo

CONFIG_FILE := config.mk
ifneq ($(wildcard $(CONFIG_FILE)),)
include $(CONFIG_FILE)
endif

check: ## run quality checks and unit tests
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types
	$(MAKE) test
	$(MAKE) rust-check

clean: ## remove cache files (Python and Rust)
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	find $(CLEAN_DIRS) -name '*.orig' -type f -delete
	cd rust && $(CARGO) clean 2>/dev/null || true

clean-env: ## remove the virtual environment directory
	rm -rf .venv

deploy: ## installs from lockfile
	git submodule update --init --recursive
	which uv || pip install --user uv
	uv sync --frozen --no-dev

init: ## pulls submodules and initializes environment (Python + Rust)
	git submodule update --init --recursive
	which uv || pip install --user uv
	uv sync --all-groups
	@echo "Building Rust CLI..."
	cd rust && $(CARGO) build --release
	@echo ""
	@echo "Setup complete! Rust CLI available at: rust/target/release/vit"

quality:
	$(MAKE) clean
	uv run ruff check $(QUALITY_DIRS)
	uv run ruff format --check $(QUALITY_DIRS)

style:
	uv run ruff check --fix $(QUALITY_DIRS)
	uv run ruff format $(QUALITY_DIRS)

test: ## run unit tests
	$(PYTHON) -m pytest \
		-rs \
		--cov=./$(PROJECT) \
		--cov-report=term \
		./tests/

test-%: ## run unit tests matching a pattern
	$(PYTHON) -m pytest -s -r fE -k $* ./tests/ --tb=no

test-pdb-%: ## run unit tests matching a pattern with PDB fallback
	$(PYTHON) -m pytest -rs --pdb -k $* -v ./tests/ 

test-ci: ## runs CI-only tests
	export "CUDA_VISIBLE_DEVICES=''" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "not ci_skip" \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		--cov-report=term \
		./tests/

types: ## run static type checking
	uv run basedpyright

# ============================================================================
# Rust targets
# ============================================================================

rust: ## build Rust CLI (debug)
	cd rust && $(CARGO) build

rust-release: ## build Rust CLI (release)
	cd rust && $(CARGO) build --release

TORCH_PATH := $(shell uv run python -c "import torch; print(torch.__path__[0])" 2>/dev/null)

rust-ffi: ## build Rust CLI with FFI/inference support (auto-detects PyTorch from virtualenv)
ifndef TORCH_PATH
	$(error PyTorch not found. Run 'make init' first or set LIBTORCH manually)
endif
	cd rust && PATH="/usr/local/cuda/bin:$(PATH)" LIBTORCH="$(TORCH_PATH)" LIBTORCH_CXX11_ABI=1 $(CARGO) build --release --features ffi
	@# Copy libraries next to binary for RPATH to work (symlinks for dev)
	@mkdir -p rust/target/release/lib
	@find rust/target/release/build -name "libvit_bridge.so" -exec cp {} rust/target/release/lib/ \;
	@for lib in $(TORCH_PATH)/lib/*.so*; do ln -sf "$$lib" rust/target/release/lib/; done
	@echo ""
	@echo "Build complete! The CLI is ready to use:"
	@echo "  ./rust/target/release/vit --help"
	@echo ""
	@echo "For a portable distribution, run: make rust-install"

rust-ffi-rocm: ## build Rust CLI with FFI/inference support for ROCm (auto-detects PyTorch from virtualenv)
ifndef TORCH_PATH
	$(error PyTorch not found. Run 'make init' first or set LIBTORCH manually)
endif
	cd rust && PATH="/usr/local/cuda/bin:$(PATH)" LIBTORCH="$(TORCH_PATH)" LIBTORCH_CXX11_ABI=1 $(CARGO) build --release --features "ffi,vit-ffi/rocm"
	@# Copy libraries next to binary for RPATH to work (symlinks for dev)
	@mkdir -p rust/target/release/lib
	@find rust/target/release/build -name "libvit_bridge.so" -exec cp {} rust/target/release/lib/ \;
	@for lib in $(TORCH_PATH)/lib/*.so*; do ln -sf "$$lib" rust/target/release/lib/; done
	@echo ""
	@echo "Build complete! The CLI is ready to use with ROCm:"
	@echo "  ./rust/target/release/vit --help"
	@echo ""
	@echo "For a portable distribution, run: make rust-install"

RUST_INSTALL_DIR ?= dist/vit

rust-install: ## create portable distribution (set RUST_INSTALL_DIR to customize)
ifndef TORCH_PATH
	$(error PyTorch not found. Run 'make init' first)
endif
	@if [ ! -f rust/target/release/vit ]; then \
		echo "Error: Build first with 'make rust-ffi'"; exit 1; \
	fi
	@echo "Creating portable distribution in $(RUST_INSTALL_DIR)..."
	@rm -rf $(RUST_INSTALL_DIR)
	@mkdir -p $(RUST_INSTALL_DIR)/lib
	@# Copy binary
	@cp rust/target/release/vit $(RUST_INSTALL_DIR)/
	@# Copy bridge library
	@find rust/target/release/build -name "libvit_bridge.so" -exec cp {} $(RUST_INSTALL_DIR)/lib/ \;
	@# Copy PyTorch libraries (resolve symlinks, only .so files)
	@for lib in $(TORCH_PATH)/lib/*.so*; do \
		cp -L "$$lib" $(RUST_INSTALL_DIR)/lib/ 2>/dev/null || true; \
	done
	@# Calculate size
	@echo ""
	@echo "Portable distribution created:"
	@echo "  $(RUST_INSTALL_DIR)/"
	@echo "  └── vit"
	@echo "  └── lib/ ($$(du -sh $(RUST_INSTALL_DIR)/lib | cut -f1))"
	@echo ""
	@echo "To use: $(RUST_INSTALL_DIR)/vit --help"
	@echo "This directory can be moved anywhere or packaged as a tarball."

rust-test: ## run Rust tests (vit-core only; use rust-test-ffi for FFI tests)
	cd rust && $(CARGO) test --package vit-core

rust-test-ffi: ## run Rust FFI tests (auto-detects PyTorch from virtualenv)
ifndef TORCH_PATH
	$(error PyTorch not found. Run 'make init' first or set LIBTORCH manually)
endif
	cd rust && PATH="/usr/local/cuda/bin:$(PATH)" LIBTORCH="$(TORCH_PATH)" LIBTORCH_CXX11_ABI=1 $(CARGO) test --package vit-ffi

rust-check: ## run Rust build and tests (vit-core only)
	cd rust && $(CARGO) build --package vit-core && $(CARGO) test --package vit-core

rust-clean: ## clean Rust build artifacts
	cd rust && $(CARGO) clean

rust-fmt: ## format Rust code
	cd rust && $(CARGO) fmt

rust-clippy: ## run Rust linter
	cd rust && $(CARGO) clippy -- -D warnings

# ============================================================================
# Export targets
# ============================================================================

export-model: ## export a ViT model to AOTInductor format (set CONFIG, WEIGHTS, OUTPUT, SHAPE, DEVICE)
ifndef CONFIG
	$(error CONFIG is not set. Example: make export-model CONFIG=config.yaml OUTPUT=model.pt2)
endif
ifndef OUTPUT
	$(error OUTPUT is not set. Example: make export-model CONFIG=config.yaml OUTPUT=model.pt2)
endif
	TORCH_COMPILE_DISABLE=1 $(PYTHON) scripts/export_aot.py \
		--config $(CONFIG) \
		$(if $(WEIGHTS),--weights $(WEIGHTS),) \
		--output $(OUTPUT) \
		--shape $(or $(SHAPE),1,3,224,224) \
		--device $(or $(DEVICE),cpu) \
		--dtype $(or $(DTYPE),float32)

help: ## display this help message
	@echo "Please use \`make <target>' where <target> is one of"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}'
