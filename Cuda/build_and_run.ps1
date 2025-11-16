# Script PowerShell para compilar e executar K-Means CUDA otimizado
# Usa NVCC com compilador MSVC

Write-Host "`n" -ForegroundColor Green
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  K-Means 1D - CUDA Otimizado          ║" -ForegroundColor Cyan
Write-Host "║  Compilação e Execução                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Configurar CUDA
$cuda_path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$nvcc_path = Join-Path $cuda_path "bin\nvcc.exe"

if (-not (Test-Path $nvcc_path)) {
    Write-Host "✗ ERRO: NVCC não encontrado em $nvcc_path" -ForegroundColor Red
    Write-Host "        Instale CUDA Toolkit 13.0 ou superior" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ NVCC encontrado: $nvcc_path" -ForegroundColor Green
Write-Host ""

# Adicionar CUDA ao PATH
$env:PATH = "$cuda_path\bin;" + $env:PATH

# Gerar dados de teste se necessário
if (-not (Test-Path "dados.csv")) {
    Write-Host "Gerando dados de teste..." -ForegroundColor Yellow
    if (Test-Path "..\generate_data.py") {
        & python "..\generate_data.py" 100000 20 42
    } else {
        Write-Host "⚠ AVISO: generate_data.py não encontrado" -ForegroundColor Yellow
    }
    Write-Host ""
}

# ===== COMPILAR VERSÃO SEQUENCIAL (CPU) =====
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  1. COMPILANDO VERSÃO SEQUENCIAL      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

& gcc -O3 -std=c99 kmeans_1d_seq.c -o kmeans_1d_seq.exe -lm
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ ERRO na compilação da versão sequencial!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Compilado: kmeans_1d_seq.exe" -ForegroundColor Green
Write-Host ""

# ===== COMPILAR VERSÃO CUDA OTIMIZADA =====
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  2. COMPILANDO VERSÃO CUDA OTIMIZADA  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "Compilando com NVCC..." -ForegroundColor Yellow
Write-Host "Arquitetura: sm_75 (GTX 1660 Ti)" -ForegroundColor Gray
Write-Host ""

& $nvcc_path -O3 -arch=sm_75 kmeans_1d_cuda_optimized.cu -o kmeans_1d_cuda_opt.exe 2>&1 | ForEach-Object {
    if ($_ -match "error") {
        Write-Host $_ -ForegroundColor Red
    } else {
        Write-Host $_
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ ERRO na compilação CUDA!" -ForegroundColor Red
    Write-Host "  Dica: Ajuste -arch se sua GPU tiver compute capability diferente" -ForegroundColor Yellow
    Write-Host "        Use 'nvidia-smi' para descobrir sua GPU" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "✓ Compilado: kmeans_1d_cuda_opt.exe" -ForegroundColor Green
Write-Host ""

# ===== EXECUTAR VERSÃO SEQUENCIAL =====
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  3. EXECUTANDO VERSÃO SEQUENCIAL      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$seq_start = Get-Date
& .\kmeans_1d_seq.exe dados.csv centroides_iniciais.csv 20
$seq_time = (Get-Date) - $seq_start
$seq_ms = $seq_time.TotalMilliseconds

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ ERRO na execução sequencial!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "⏱  Tempo total (CPU): $([math]::Round($seq_ms, 2)) ms" -ForegroundColor Cyan
Write-Host ""

# ===== EXECUTAR VERSÃO CUDA OTIMIZADA =====
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  4. EXECUTANDO VERSÃO CUDA OTIMIZADA  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Parâmetros: K=20, max_iter=100, eps=1e-6" -ForegroundColor Gray
Write-Host ""

$cuda_start = Get-Date
& .\kmeans_1d_cuda_opt.exe dados.csv centroides_iniciais.csv 20 100 1e-6
$cuda_time = (Get-Date) - $cuda_start
$cuda_ms = $cuda_time.TotalMilliseconds

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ ERRO na execução CUDA!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "⏱  Tempo total (GPU): $([math]::Round($cuda_ms, 2)) ms" -ForegroundColor Cyan
Write-Host ""

# ===== ANÁLISE DE DESEMPENHO =====
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  5. ANÁLISE DE DESEMPENHO             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$speedup = $seq_ms / $cuda_ms
$speedup_text = [math]::Round($speedup, 2)
$percent_faster = (($seq_ms - $cuda_ms) / $seq_ms) * 100

Write-Host "Tempo CPU:                 $([math]::Round($seq_ms, 2)) ms" -ForegroundColor White
Write-Host "Tempo GPU:                 $([math]::Round($cuda_ms, 2)) ms" -ForegroundColor White
Write-Host ""

if ($speedup -gt 1) {
    Write-Host "Speedup (CPU/GPU):         ${speedup_text}x" -ForegroundColor Green
    Write-Host "GPU é $([math]::Round($percent_faster, 1))% mais rápida que CPU" -ForegroundColor Green
} else {
    $slowdown = 1 / $speedup
    Write-Host "Speedup (CPU/GPU):         $($speedup_text)x" -ForegroundColor Yellow
    Write-Host "CPU é $([math]::Round($percent_faster * -1, 1))% mais rápida que GPU" -ForegroundColor Yellow
    Write-Host "(Normal para overhead CUDA em datasets pequenos)" -ForegroundColor Gray
}

Write-Host ""

# ===== VALIDAR RESULTADOS =====
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  6. VALIDANDO RESULTADOS              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "compare_cuda_results.py") {
    & python compare_cuda_results.py
} else {
    Write-Host "⚠ Script de validação não encontrado" -ForegroundColor Yellow
    Write-Host "  Verifique manualmente:" -ForegroundColor Gray
    Write-Host "    - assign_seq.csv vs assign_cuda.csv" -ForegroundColor Gray
    Write-Host "    - centroids_seq.csv vs centroids_cuda.csv" -ForegroundColor Gray
}

Write-Host ""

# ===== RESUMO FINAL =====
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  COMPILAÇÃO E EXECUÇÃO CONCLUÍDA!     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Executáveis gerados:" -ForegroundColor Yellow
Write-Host "  • kmeans_1d_seq.exe        (CPU sequencial)" -ForegroundColor Gray
Write-Host "  • kmeans_1d_cuda_opt.exe   (GPU CUDA otimizado)" -ForegroundColor Gray
Write-Host ""
Write-Host "Resultados:" -ForegroundColor Yellow
Write-Host "  • assign_seq.csv           (atribuições CPU)" -ForegroundColor Gray
Write-Host "  • assign_cuda.csv          (atribuições GPU)" -ForegroundColor Gray
Write-Host "  • centroids_seq.csv        (centróides CPU)" -ForegroundColor Gray
Write-Host "  • centroids_cuda.csv       (centróides GPU)" -ForegroundColor Gray
Write-Host "  • metrics_cuda.txt         (métricas de desempenho)" -ForegroundColor Gray
Write-Host ""
Write-Host "Resumo de desempenho:" -ForegroundColor Yellow
Write-Host "  CPU: $([math]::Round($seq_ms, 2)) ms" -ForegroundColor Gray
Write-Host "  GPU: $([math]::Round($cuda_ms, 2)) ms" -ForegroundColor Gray
Write-Host "  Speedup: $($speedup_text)x" -ForegroundColor Cyan
Write-Host ""
