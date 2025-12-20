Write-Host "Testing Credit Risk API Version 2" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Wait for API to be ready
Write-Host "Waiting for API to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    try {
        $health = Invoke-RestMethod http://localhost:8001/health -TimeoutSec 2
        Write-Host "✓ API is ready!" -ForegroundColor Green
        break
    } catch {
        $attempt++
        Write-Host "  Attempt $attempt/$maxAttempts: API not ready yet..." -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
}

if ($attempt -eq $maxAttempts) {
    Write-Host "✗ API failed to start within $maxAttempts attempts" -ForegroundColor Red
    exit 1
}

# Test endpoints
Write-Host "`nTesting endpoints:" -ForegroundColor Yellow

$endpoints = @(
    @{Name = "Root"; Method = "GET"; Path = "/"},
    @{Name = "Health"; Method = "GET"; Path = "/health"},
    @{Name = "Model Info"; Method = "GET"; Path = "/model-info"}
)

foreach ($endpoint in $endpoints) {
    Write-Host "  Testing $($endpoint.Name)..." -NoNewline
    try {
        $response = Invoke-WebRequest "http://localhost:8001$($endpoint.Path)" -Method $endpoint.Method -TimeoutSec 5
        Write-Host " ✓ (Status: $($response.StatusCode))" -ForegroundColor Green
    } catch {
        Write-Host " ✗ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Test prediction
Write-Host "`nTesting prediction:" -ForegroundColor Yellow

$payload = @{
    customer_id = "V2_TEST_001"
    features = @{
        Year_mean = 2019.0
        Month_mean = 8.0
    }
} | ConvertTo-Json

Write-Host "  Sending prediction request..." -NoNewline
try {
    $response = Invoke-RestMethod http://localhost:8001/predict -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 10
    Write-Host " ✓ Success!" -ForegroundColor Green
    Write-Host "`nPrediction Result:" -ForegroundColor White
    $response | Format-List | Out-String | Write-Host
    
    # Check for new fields
    $hasNewFields = $response.risk_category -and $response.risk_score -and $response.recommendation
    if ($hasNewFields) {
        Write-Host "✓ All new fields present (risk_category, risk_score, recommendation)" -ForegroundColor Green
    }
} catch {
    Write-Host " ✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nAPI Version 2 Test Complete!" -ForegroundColor Green
Write-Host "`nAccess:" -ForegroundColor Cyan
Write-Host "  API: http://localhost:8001" -ForegroundColor White
Write-Host "  Docs: http://localhost:8001/docs" -ForegroundColor White
Write-Host "  MLflow: http://localhost:5001" -ForegroundColor White