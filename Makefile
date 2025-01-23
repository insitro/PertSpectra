ENV_NAME=pertspectra
CONDA_BASE=$(shell conda run -n base conda  info --base)
ENV_DIR=$(CONDA_BASE)/envs/$(ENV_NAME)


## Installation
.PHONY: check-conda
check-conda:
ifeq (,$(shell which conda))
	$(error "This project uses conda for environment management. Please install conda before continuing.")
endif

check-conda-lock:
ifeq  (,$(shell which conda-lock))
	$(error "conda-lock command not found, install with `pip install conda-lock`")
endif

.PHONY: conda-lock
conda-lock: check-conda
	pip install conda-lock && \
	conda-lock lock -f environment.yaml -p osx-64 -p linux-64

.PHONY: install-env
install-env: check-conda check-conda-lock
	conda-lock install -p $(ENV_DIR) conda-lock.yml && \
	conda run -p $(ENV_DIR) python -m pip install -r requirements.txt

.PHONY: install-pre-commit
install-pre-commit:
	conda run -p $(ENV_DIR) pre-commit install

.PHONY: jupyter-kernel
jupyter-kernel:
	conda run -p $(ENV_DIR) $(ENV_DIR)/bin/pip install ipykernel
	conda run -p $(ENV_DIR) $(ENV_DIR)/bin/ipython kernel install --user --name=$(ENV_NAME)

install: install-env install-pre-commit jupyter-kernel

## Linting

.PHONY: format
format:
	ruff check --fix

.PHONY: type-check
type-check:
	pre-commit run mypy --all-files

.PHONY: lint
lint:
	pre-commit run --all-files


## Testing
COVERAGE_REPORT_FILE=coverage.xml
PYTEST_REPORT_FILE=report.xml

.PHONY: pytest
pytest:
	pytest -v --junitxml $(PYTEST_REPORT_FILE) $(ENV_NAME)

.PHONY: pytest-cov
pytest-cov:
	pytest --cov $(PACKAGE_NAME) $(ENV_NAME)
