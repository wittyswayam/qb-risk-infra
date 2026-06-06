# syntax=docker/dockerfile:1.6
# ============================================================
# Stage 1: Build dependencies
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# System dependencies needed at build time only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a separate prefix so we can copy
# only the site-packages layer into the final stage
COPY requirements.txt .
RUN pip install --upgrade pip==24.0 && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM python:3.12-slim AS runtime

LABEL maintainer="samrat swayam <kumarsamrat408@gmail.com>"
LABEL org.opencontainers.image.title="qb-risk-infra"
LABEL org.opencontainers.image.description="Quantitative Backtesting & Risk Analytics API"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Runtime system libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Copy application source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

# Create required directories
RUN mkdir -p logs data/raw && chown -R appuser:appuser logs data

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/ping || exit 1

EXPOSE 8000

# Entrypoint: uvicorn with production settings
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log", \
     "--no-server-header"]
