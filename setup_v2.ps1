Write-Host "Setting up Credit Risk API Version 2" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Create directory structure
$directories = @(
    "src_2",
    "src_2/api",
    "mlflow_v2",
    "minio_data_v2",
    "postgres_data_v2"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Copy source files if they exist
$filesToCopy = @(
    @{Source = "main_2.py"; Destination = "src_2/api/main_2.py"},
    @{Source = "predictor_2.py"; Destination = "src_2/api/predictor_2.py"},
    @{Source = "pydantic_models_2.py"; Destination = "src_2/api/pydantic_models_2.py"}
)

foreach ($file in $filesToCopy) {
    if (Test-Path $file.Source) {
        Copy-Item -Path $file.Source -Destination $file.Destination -Force
        Write-Host "Copied: $($file.Source) -> $($file.Destination)" -ForegroundColor Green
    } else {
        Write-Host "Warning: Source file not found: $($file.Source)" -ForegroundColor Yellow
    }
}

# Create __init__.py files
$initFiles = @(
    "src_2/__init__.py",
    "src_2/api/__init__.py"
)

foreach ($initFile in $initFiles) {
    if (-not (Test-Path $initFile)) {
        "" | Out-File -FilePath $initFile -Encoding UTF8
        Write-Host "Created: $initFile" -ForegroundColor Green
    }
}

# Create requirements_2.txt if it doesn't exist
if (-not (Test-Path "requirements_2.txt")) {
    @"
# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Machine Learning
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.26.2
joblib==1.3.2

# MLflow for model registry
mlflow==2.9.2

# Testing
pytest==7.4.3
pytest-cov==4.1.0
httpx==0.25.1

# Code quality
flake8==6.1.0
black==23.11.0

# Utilities
python-multipart==0.0.6
"@ | Out-File -FilePath "requirements_2.txt" -Encoding UTF8
    Write-Host "Created: requirements_2.txt" -ForegroundColor Green
}

# Ensure feature_names.json exists
if (-not (Test-Path "models/feature_names.json")) {
    @"
[
  "Year_mean",
  "Month_mean"
]
"@ | Out-File -FilePath "models/feature_names.json" -Encoding UTF8
    Write-Host "Created: models/feature_names.json" -ForegroundColor Green
}

Write-Host "`nSetup complete! You can now:" -ForegroundColor Cyan
Write-Host "1. Build and run: docker-compose -f docker-compose_2.yml up --build" -ForegroundColor White
Write-Host "2. Test API: curl http://localhost:8001/health" -ForegroundColor White
Write-Host "3. Access MLflow UI: http://localhost:5001" -ForegroundColor White
Write-Host "4. Access MinIO console: http://localhost:9002" -ForegroundColor White