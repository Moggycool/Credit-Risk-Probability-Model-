Write-Host "Building Credit Risk API Version 2" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Check if models directory exists
Write-Host "`nChecking directory structure..." -ForegroundColor Yellow

if (-not (Test-Path "models")) {
    Write-Host "✗ models/ directory not found!" -ForegroundColor Red
    Write-Host "Creating models/ directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "models" -Force
}

# List contents of models directory
Write-Host "`nContents of models/ directory:" -ForegroundColor Green
Get-ChildItem -Path "models" | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor Gray
}

# Verify key files exist
Write-Host "`nVerifying required model files:" -ForegroundColor Yellow

$requiredModels = @(
    "logistic_champion_fixed.joblib",
    "feature_names.json"
)

$allModelsExist = $true
foreach ($model in $requiredModels) {
    $modelPath = "models/$model"
    if (Test-Path $modelPath) {
        Write-Host "  ✓ $model" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $model (MISSING)" -ForegroundColor Red
        $allModelsExist = $false
    }
}

if (-not $allModelsExist) {
    Write-Host "`nSome model files are missing. Creating placeholder..." -ForegroundColor Yellow
    
    # Create feature_names.json if missing
    if (-not (Test-Path "models/feature_names.json")) {
        @"
[
  "Year_mean",
  "Month_mean"
]
"@ | Out-File -FilePath "models/feature_names.json" -Encoding UTF8
        Write-Host "  Created placeholder feature_names.json" -ForegroundColor Green
    }
    
    # Check if any model file exists
    $modelFiles = Get-ChildItem -Path "models" -Filter "*.joblib" -ErrorAction SilentlyContinue
    if ($modelFiles.Count -eq 0) {
        Write-Host "  No .joblib model files found in models/" -ForegroundColor Yellow
        Write-Host "  Using volume mount from host instead" -ForegroundColor Yellow
    }
}

# Build Docker image
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
try {
    # Use docker build directly for more control
    docker build -f Dockerfile_2 -t credit-risk-api-v2 .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker build successful!" -ForegroundColor Green
    } else {
        Write-Host "✗ Docker build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ Docker build error: $_" -ForegroundColor Red
    exit 1
}

# Run with docker-compose
Write-Host "`nStarting services with docker-compose..." -ForegroundColor Yellow
try {
    # Run in background
    docker-compose -f docker_compose_2.yml up -d
    
    Write-Host "✓ Services started!" -ForegroundColor Green
    Write-Host "`nContainer status:" -ForegroundColor Cyan
    docker-compose -f docker_compose_2.yml ps
    
    Write-Host "`nAPI will be available at:" -ForegroundColor Cyan
    Write-Host "  http://localhost:8001" -ForegroundColor White
    Write-Host "  Docs: http://localhost:8001/docs" -ForegroundColor White
    
    Write-Host "`nChecking API health in 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    try {
        $health = Invoke-RestMethod http://localhost:8001/health -TimeoutSec 5
        Write-Host "✓ API is healthy!" -ForegroundColor Green
        Write-Host "  Model loaded: $($health.model_loaded)" -ForegroundColor Green
        Write-Host "  Model source: $($health.model_source)" -ForegroundColor Green
    } catch {
        Write-Host "✗ API not responding yet. Check logs with:" -ForegroundColor Yellow
        Write-Host "  docker-compose -f docker_compose_2.yml logs" -ForegroundColor White
    }
    
} catch {
    Write-Host "✗ Failed to start services: $_" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check if port 8001 is already in use" -ForegroundColor White
    Write-Host "2. Check Docker logs: docker-compose -f docker_compose_2.yml logs" -ForegroundColor White
    Write-Host "3. Try running without -d flag: docker-compose -f docker_compose_2.yml up" -ForegroundColor White
}