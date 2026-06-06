# ============================================================
# QB Risk Infra — Makefile
# ============================================================
# Usage:
#   make install       - install production dependencies
#   make install-dev   - install development dependencies
#   make test          - run full test suite
#   make lint          - run ruff + black check
#   make format        - auto-format with black
#   make api           - run the FastAPI server locally
#   make dashboard     - run the Streamlit dashboard
#   make docker-up     - start Docker Compose stack
#   make docker-down   - stop Docker Compose stack
#   make clean         - remove build artefacts
# ============================================================

.PHONY: install install-dev test lint format type-check api dashboard \
        docker-build docker-up docker-down docker-logs migrate clean help

PYTHON     := python3
PIP        := $(PYTHON) -m pip
PYTEST     := $(PYTHON) -m pytest
BLACK      := $(PYTHON) -m black
RUFF       := $(PYTHON) -m ruff
MYPY       := $(PYTHON) -m mypy
UVICORN    := $(PYTHON) -m uvicorn
STREAMLIT  := $(PYTHON) -m streamlit

SRC_DIRS   := src tests
COVERAGE   := htmlcov

# ---- Dependencies ----------------------------------------------------------

install:
	@echo "Installing production dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PYTHON) -m pre_commit install

# ---- Code quality ----------------------------------------------------------

format:
	@echo "Formatting with black..."
	$(BLACK) $(SRC_DIRS)

lint:
	@echo "Linting with ruff..."
	$(RUFF) check $(SRC_DIRS)
	@echo "Checking formatting with black..."
	$(BLACK) --check $(SRC_DIRS)

type-check:
	@echo "Type checking with mypy..."
	$(MYPY) src --ignore-missing-imports

check: lint type-check
	@echo "All checks passed."

# ---- Testing ---------------------------------------------------------------

test:
	@echo "Running test suite..."
	$(PYTEST) tests/unit/ -v --tb=short --cov=src --cov-report=term-missing

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v -m integration

test-fast:
	$(PYTEST) tests/unit/ -v --tb=short -m "not slow" -q

test-cov-html:
	$(PYTEST) tests/ --cov=src --cov-report=html:$(COVERAGE)
	@echo "Coverage report: $(COVERAGE)/index.html"

# ---- Application -----------------------------------------------------------

api:
	@echo "Starting API server (development mode)..."
	QB_CONFIG=config/config.yaml \
	$(UVICORN) src.api.main:app \
		--host 0.0.0.0 --port 8000 --reload \
		--log-level debug

dashboard:
	@echo "Starting Streamlit dashboard..."
	$(STREAMLIT) run src/dashboard/app.py \
		--server.port 8501 --server.address 0.0.0.0

# ---- Docker ----------------------------------------------------------------

docker-build:
	@echo "Building Docker image..."
	docker build -t qb-risk-infra:latest .

docker-up:
	@echo "Starting Docker Compose stack..."
	docker compose up -d
	docker compose ps

docker-down:
	@echo "Stopping Docker Compose stack..."
	docker compose down

docker-down-v:
	@echo "Stopping Docker Compose stack and removing volumes..."
	docker compose down -v

docker-logs:
	docker compose logs -f api

docker-shell:
	docker compose exec api bash

# ---- Database --------------------------------------------------------------

create-tables:
	@echo "Creating database tables..."
	$(PYTHON) -c "from src.db.session import create_all_tables; create_all_tables()"

# ---- Setup -----------------------------------------------------------------

setup-data-dir:
	@echo "Creating data directories..."
	mkdir -p data/raw data/processed logs

setup: setup-data-dir create-tables
	@echo "Setup complete."

# ---- Pre-commit ------------------------------------------------------------

pre-commit:
	$(PYTHON) -m pre_commit run --all-files

# ---- Clean -----------------------------------------------------------------

clean:
	@echo "Cleaning build artefacts..."
	rm -rf $(COVERAGE) .coverage .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete."

# ---- Help ------------------------------------------------------------------

help:
	@echo ""
	@echo "QB Risk Infra — Available Make Targets"
	@echo "=========================================="
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies + pre-commit hooks"
	@echo "  format           Auto-format source code with black"
	@echo "  lint             Lint with ruff + black check"
	@echo "  type-check       Type check with mypy"
	@echo "  test             Run full unit test suite with coverage"
	@echo "  test-fast        Run tests excluding slow markers"
	@echo "  test-cov-html    Generate HTML coverage report"
	@echo "  api              Start FastAPI development server"
	@echo "  dashboard        Start Streamlit dashboard"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-up        Start full Docker Compose stack"
	@echo "  docker-down      Stop Docker Compose stack"
	@echo "  docker-logs      Tail API container logs"
	@echo "  create-tables    Create DB schema"
	@echo "  clean            Remove build/cache artefacts"
	@echo ""
