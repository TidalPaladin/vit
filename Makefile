.PHONY: clean clean-env check quality style tag-version test env upload upload-test

PROJECT=vit
QUALITY_DIRS=$(PROJECT) tests benchmark
CLEAN_DIRS=$(PROJECT) tests benchmark
UV_VERSION=0.11.28
UVX=uvx
UV=$(UVX) --from uv==$(UV_VERSION) uv
PYTHON=$(UV) run python

CONFIG_FILE := config.mk
ifneq ($(wildcard $(CONFIG_FILE)),)
include $(CONFIG_FILE)
endif

check: ## run quality checks and unit tests
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types
	$(MAKE) test

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
	export "CUDA_VISIBLE_DEVICES=''" && \
	$(PYTHON) -m pytest \
		-rs \
		-m "not cuda and not compile" \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		--cov-report=term \
		./tests/

types: ## run static type checking
	$(UV) run basedpyright

help: ## display this help message
	@echo "Please use \`make <target>' where <target> is one of"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}'
