# Multi-stage Dockerfile (build wheels, then install in small runtime image)
# Uses Python 3.11-slim
FROM python:3.12-slim as builder

WORKDIR /app

# Install build tools needed to build wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
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

# Copy application code
COPY src /app/src
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Do NOT copy models (we mount them at runtime) to keep the image small.
EXPOSE 8000

# Healthcheck uses curl - ensure curl present
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the app (uvicorn). For production you can replace this with gunicorn + uvicorn workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]