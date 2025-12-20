Write-Host "Running Both API Versions" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

Write-Host "`nVersion 1 (Original):" -ForegroundColor Yellow
Write-Host "  Port: 8000" -ForegroundColor White
Write-Host "  MLflow: 5000" -ForegroundColor White
Write-Host "  Model: Local .joblib file" -ForegroundColor White

Write-Host "`nVersion 2 (Updated):" -ForegroundColor Yellow
Write-Host "  Port: 8001" -ForegroundColor White
Write-Host "  MLflow: 5001" -ForegroundColor White
Write-Host "  Model: MLflow Registry + Enhanced features" -ForegroundColor White

Write-Host "`nStarting both versions..." -ForegroundColor Green

# Start V1 in background
Write-Host "`nStarting Version 1..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "docker-compose up"
Start-Sleep -Seconds 5

# Start V2 in background
Write-Host "`nStarting Version 2..." -ForegroundColor Gray
Start-Process powershell -ArgumentList "-NoExit", "-Command", "docker-compose -f docker-compose_2.yml up"
Start-Sleep -Seconds 5

Write-Host "`nBoth versions running!" -ForegroundColor Green
Write-Host "`nTest URLs:" -ForegroundColor Cyan
Write-Host "  V1 API: http://localhost:8000" -ForegroundColor White
Write-Host "  V1 Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  V2 API: http://localhost:8001" -ForegroundColor White
Write-Host "  V2 Docs: http://localhost:8001/docs" -ForegroundColor White
Write-Host "  V2 MLflow: http://localhost:5001" -ForegroundColor White

Write-Host "`nTo stop both:" -ForegroundColor Yellow
Write-Host "  docker-compose down" -ForegroundColor White
Write-Host "  docker-compose -f docker-compose_2.yml down" -ForegroundColor White