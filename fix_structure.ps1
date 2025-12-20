Write-Host "Fixing file structure for Docker..." -ForegroundColor Cyan

# Rename files in src_2/api to match Docker expectations
$apiPath = "src_2/api"

Write-Host "Renaming files in $apiPath..." -ForegroundColor Yellow

# Rename files (remove _2 suffix)
Rename-Item -Path "$apiPath/main_2.py" -NewName "main.py" -Force
Rename-Item -Path "$apiPath/predictor_2.py" -NewName "predictor.py" -Force
Rename-Item -Path "$apiPath/pydantic_models_2.py" -NewName "pydantic_models.py" -Force

Write-Host "✓ Files renamed:" -ForegroundColor Green
Get-ChildItem -Path $apiPath | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor Gray
}

# Create requirements_2.txt if it doesn't exist
if (-not (Test-Path "requirements_2.txt")) {
    Write-Host "`nCreating requirements_2.txt..." -ForegroundColor Yellow
    
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
boto3==1.34.0

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
    
    Write-Host "✓ Created requirements_2.txt" -ForegroundColor Green
}

Write-Host "`nStructure is now Docker-ready!" -ForegroundColor Green
Write-Host "`nBuild command:" -ForegroundColor Yellow
Write-Host "docker-compose -f docker_compose_2.yml up --build" -ForegroundColor White