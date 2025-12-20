Write-Host "Testing Docker Build for Version 2" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Check if files exist
Write-Host "`nChecking required files:" -ForegroundColor Yellow

$requiredFiles = @(
    "Dockerfile_2",
    "docker_compose_2.yml",
    "requirements_2.txt",
    "src_2/api/main.py",
    "src_2/api/predictor.py", 
    "src_2/api/pydantic_models.py",
    "models/logistic_champion_fixed.joblib",
    "models/feature_names.json"
)

$allExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (MISSING)" -ForegroundColor Red
        $allExist = $false
    }
}

if (-not $allExist) {
    Write-Host "`nSome files are missing. Run fix_structure.ps1 first." -ForegroundColor Red
    exit 1
}

# Build Docker image
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
try {
    docker-compose -f docker_compose_2.yml build
    Write-Host "✓ Build successful!" -ForegroundColor Green
} catch {
    Write-Host "✗ Build failed: $_" -ForegroundColor Red
    exit 1
}

# Run container
Write-Host "`nStarting container..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "docker-compose -f docker_compose_2.yml up"

Write-Host "`nContainer starting on port 8001..." -ForegroundColor Green
Write-Host "Wait 30 seconds, then test with:" -ForegroundColor Cyan
Write-Host "  curl http://localhost:8001/health" -ForegroundColor White
Write-Host "  or run: .\test_v2_api.ps1" -ForegroundColor White