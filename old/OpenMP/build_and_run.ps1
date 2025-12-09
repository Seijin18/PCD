# Build and Run OpenMP K-Means

Write-Host ""
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  K-Means 1D OpenMP - Build e Testes" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Diretório de trabalho
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Remover executáveis antigos
Write-Host "🗑️  Limpando executáveis antigos..." -ForegroundColor Yellow
Remove-Item "kmeans_1d_serial.exe" -ErrorAction SilentlyContinue
Remove-Item "kmeans_1d_omp.exe" -ErrorAction SilentlyContinue
Write-Host ""

# Compilação
Write-Host "🔨 Compilando versão sequencial..." -ForegroundColor Cyan
gcc -O3 -std=c99 -lm kmeans_1d_serial.c -o kmeans_1d_serial.exe
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ kmeans_1d_serial.exe compilado com sucesso" -ForegroundColor Green
} else {
    Write-Host "   ❌ Erro na compilação sequencial" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "🔨 Compilando versão OpenMP..." -ForegroundColor Cyan
gcc -O3 -std=c99 -fopenmp -lm kmeans_1d_omp.c -o kmeans_1d_omp.exe
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ kmeans_1d_omp.exe compilado com sucesso" -ForegroundColor Green
} else {
    Write-Host "   ❌ Erro na compilação OpenMP" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Testes
Write-Host "🧪 EXECUTANDO TESTES" -ForegroundColor Cyan
Write-Host ""

Write-Host "1️⃣  Teste Sequencial (baseline):" -ForegroundColor Yellow
$seqStart = Get-Date
.\kmeans_1d_serial.exe dados.csv centroides_iniciais.csv 20 100 1e-6 | Out-Null
$seqEnd = Get-Date
$seqTime = ($seqEnd - $seqStart).TotalMilliseconds
Write-Host "   ✅ Tempo: $([Math]::Round($seqTime, 2))ms" -ForegroundColor Green
Write-Host ""

# Testar OpenMP com diferentes números de threads
$thread_counts = @(1, 2, 4, 8, 16)

foreach ($threads in $thread_counts) {
    Write-Host "$($threads)️⃣  Teste OpenMP com $threads threads:" -ForegroundColor Yellow
    $env:OMP_NUM_THREADS = $threads
    
    $ompStart = Get-Date
    .\kmeans_1d_omp.exe dados.csv centroides_iniciais.csv 20 100 1e-6 | Out-Null
    $ompEnd = Get-Date
    $ompTime = ($ompEnd - $ompStart).TotalMilliseconds
    
    $speedup = [Math]::Round($seqTime / $ompTime, 2)
    Write-Host "   ✅ Tempo: $([Math]::Round($ompTime, 2))ms (Speedup: ${speedup}x)" -ForegroundColor Green
    Write-Host ""
}

Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ TESTES CONCLUÍDOS" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
