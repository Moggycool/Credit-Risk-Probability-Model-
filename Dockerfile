# Multi-stage Dockerfile (build wheels, then install in small runtime image)
# Uses Python 3.12-slim (pin a patch version if you want reproducible builds)
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools needed to build wheels.
# Add extra -dev packages only if your requirements need them (libssl-dev, libffi-dev, libpq-dev, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libssl-dev libffi-dev \
    # libpq-dev default-libmysqlclient-dev libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements and build wheels
COPY requirements-prod.txt /app/requirements-prod.txt
RUN python -m pip install --upgrade pip wheel setuptools \
    && python -m pip wheel --no-cache-dir --wheel-dir /wheels -r /app/requirements-prod.txt

# Runtime image
FROM python:3.12-slim

WORKDIR /app

# Minimal runtime deps: curl used for healthcheck, ca-certificates for TLS
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder and install
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Copy application code (keep as /app/src so PYTHONPATH works)
COPY src /app/src
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Create a non-root user for runtime (safer than running as root)
RUN useradd --create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# Do NOT copy models (we mount them at runtime) to keep the image small.
EXPOSE 8000

# Healthcheck uses curl - ensure curl present
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the app (uvicorn). For production consider gunicorn + uvicorn workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]