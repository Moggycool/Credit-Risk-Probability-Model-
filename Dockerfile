FROM python:3.11-slim

WORKDIR /app

# Minimal system deps for building wheels (scikit-learn, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Use production requirements
COPY requirements-prod.txt /app/requirements-prod.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-prod.txt

# Copy app code
COPY src /app/src
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]