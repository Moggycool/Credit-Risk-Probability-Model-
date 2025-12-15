# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY src /app/src
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Runtime environment variables can be set at docker-compose / container runtime:
# MODEL_NAME, MODEL_STAGE, MODEL_LOCAL_PATH, MLFLOW_TRACKING_URI

# Start Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]