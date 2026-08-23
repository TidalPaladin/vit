.PHONY: audit-dependencies audit-workflows benchmark-packed-cuda check check-distribution clean clean-env deploy init quality report-deprecations style test test-ci test-compile-cpu test-compile-cuda test-deprecations test-packed-cuda types

PROJECT=vit
QUALITY_DIRS=$(PROJECT) tests benchmark tools examples
CLEAN_DIRS=$(PROJECT) tests benchmark tools examples
UV_VERSION=0.11.28
UVX=uvx
UV=$(UVX) --from uv==$(UV_VERSION) uv
PYTHON=$(UV) run python
PYTHON_VERSION?=3.14
REPORT_DIR?=dependency_reports
PIP_AUDIT=$(UV) run --isolated --frozen --only-group ci-security --python 3.14 pip-audit

CONFIG_FILE := config.mk
ifneq ($(wildcard $(CONFIG_FILE)),)
include $(CONFIG_FILE)
endif

check: ## run quality checks and unit tests
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types
	$(MAKE) check-distribution
	$(MAKE) test

audit-dependencies: ## audit locked Python dependencies for known public vulnerabilities
	@set -e; \
	audit_directory="$$(mktemp -d)"; \
	requirements_file="$$audit_directory/requirements.txt"; \
	trap 'rm -rf "$$audit_directory"' EXIT; \
	$(UV) export \
		--frozen \
		--all-groups \
		--no-emit-project \
		--quiet \
		--python "$(PYTHON_VERSION)" \
		--output-file "$$requirements_file"; \
	$(PIP_AUDIT) \
		--strict \
		--disable-pip \
		--require-hashes \
		--cache-dir "$$audit_directory/cache" \
		--progress-spinner=off \
		--vulnerability-service pypi \
		--requirement "$$requirements_file" \
		$(if $(AUDIT_REPORT),--format json --output "$(AUDIT_REPORT)")

audit-workflows: ## audit GitHub Actions workflows with locked zizmor
	$(UV) run --isolated --frozen --only-group ci-security --python 3.14 zizmor \
		--strict-collection \
		--pedantic \
		--offline \
		--min-severity low \
		--format json \
		.github/workflows \
		$(if $(ZIZMOR_REPORT),> "$(ZIZMOR_REPORT)")

check-distribution: ## build a wheel and validate its console-script targets
	@dist_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$dist_dir"' EXIT; \
	$(UV) build --wheel --out-dir "$$dist_dir"; \
	$(PYTHON) tools/validate_wheel.py "$$dist_dir"/*.whl

clean: ## remove cache files
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	find $(CLEAN_DIRS) -name '*.orig' -type f -delete

clean-env: ## remove the virtual environment directory
	rm -rf .venv


deploy: ## installs from lockfile
	git submodule update --init --recursive
	which $(UVX) || python -m pip install --user "uv==$(UV_VERSION)"
	$(UV) sync --frozen --no-dev


init: ## pulls submodules and initializes virtual environment
	git submodule update --init --recursive
	which $(UVX) || python -m pip install --user "uv==$(UV_VERSION)"
	$(UV) sync --all-groups

quality:
	$(MAKE) clean
	$(UV) run ruff check $(QUALITY_DIRS)
	$(UV) run ruff format --check $(QUALITY_DIRS)

style:
	$(UV) run ruff check --fix $(QUALITY_DIRS)
	$(UV) run ruff format $(QUALITY_DIRS)

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

test-ci: ## runs CI-only tests (excludes cuda and compile tests)
	export CUDA_VISIBLE_DEVICES="" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "not cuda and not compile" \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		--cov-report=term \
		./tests/

test-compile-cpu: ## run CPU-only torch.compile tests with Dynamo enabled
	export CUDA_VISIBLE_DEVICES="" TORCHDYNAMO_DISABLE="0" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "compile and not cuda" \
		./tests/

test-compile-cuda: ## run CUDA-only torch.compile tests with Dynamo enabled
	export TORCHDYNAMO_DISABLE="0" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "compile and cuda" \
		./tests/

test-packed-cuda: ## run packed variable-length CUDA tests
	export TORCHDYNAMO_DISABLE="1" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "cuda and not compile" \
		./tests/test_packed.py
	export TORCHDYNAMO_DISABLE="0" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "cuda and compile" \
		./tests/test_packed.py

benchmark-packed-cuda: ## run the three-pass packed attention decision benchmark
	$(UV) run vit-packed-attention-benchmark --independent-runs 3

report-deprecations: ## report direct dependency yanks, inactivity, and Python conflicts
	$(UV) run --isolated --frozen --only-group ci-dependency-report --python 3.14 python tools/dependency_report.py \
		--json-output "$(REPORT_DIR)/dependency-deprecations.json" \
		--summary-output "$(REPORT_DIR)/dependency-deprecations.md"

test-deprecations: ## run CPU tests with default deprecation warnings
	export CUDA_VISIBLE_DEVICES="" TORCHDYNAMO_DISABLE="1" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "not cuda and not compile" \
		-W default \
		./tests/

types: ## run static type checking
	$(UV) run basedpyright

help: ## display this help message
	@echo "Please use \`make <target>' where <target> is one of"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}'
