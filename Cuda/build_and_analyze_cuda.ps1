Write-Host "======================================================================" -ForegroundColor Magenta
Write-Host "         K-Means CUDA Otimizado - Build and Analysis               " -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Magenta
Write-Host ""

# Definir diretorio
Push-Location "d:\Projetinhos\Faculdade\PCD\Entrega 1\Cuda"

# Verificar CUDA
Write-Host "Checking CUDA..." -ForegroundColor Cyan
$gpu_info = nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i 0
Write-Host "GPU: $gpu_info" -ForegroundColor Green

# Detectar compute capability
$cap_output = nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0
$cap = $cap_output.Trim().Replace(" ", "").Replace(".", "")
$Architecture = "sm_" + $cap
Write-Host "Compute Capability: $Architecture" -ForegroundColor Green
Write-Host ""

# Compilar CUDA
Write-Host "Compiling CUDA code with $Architecture..." -ForegroundColor Magenta
$nvcc_cmd = "nvcc.exe -O3 -arch=$Architecture kmeans_1d_cuda_optimized.cu -o kmeans_1d_cuda_opt.exe"
Write-Host "Command: $nvcc_cmd" -ForegroundColor Cyan
Write-Host ""

Invoke-Expression $nvcc_cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation successful!" -ForegroundColor Green
} else {
    Write-Host "Compilation failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Host ""

# Executar teste
Write-Host "Running K-Means CUDA test..." -ForegroundColor Magenta
Write-Host "Parameters: K=20, max_iter=100, epsilon=1e-6" -ForegroundColor Cyan
Write-Host ""

.\kmeans_1d_cuda_opt.exe data/dados.csv data/centroides_iniciais.csv 20 100 1e-6

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Execution failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host ""
Write-Host "Checking generated files..." -ForegroundColor Cyan
Get-Item -Path results/*.csv, results/*.txt -ErrorAction SilentlyContinue | Select-Object Name, @{Name="Size(bytes)";Expression={$_.Length}} | Format-Table
Write-Host ""

# Gerar graficos se Python disponivel
Write-Host "Checking for Python..." -ForegroundColor Cyan
$python_exists = $false
try {
    $python_version = python --version
    Write-Host "Python found: $python_version" -ForegroundColor Green
    $python_exists = $true
} catch {
    Write-Host "Python not found" -ForegroundColor Yellow
}

if ($python_exists) {
    if (Test-Path "generate_performance_graphs.py") {
        Write-Host ""
        Write-Host "Generating performance graphs..." -ForegroundColor Magenta
        python generate_performance_graphs.py .
        
        if (Test-Path "graphs") {
            Write-Host ""
            Write-Host "Graphs generated:" -ForegroundColor Green
            Get-ChildItem "graphs" -Filter "*.png" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  - $($_.Name)" }
        }
        
        if (Test-Path "ANALISE_DESEMPENHO.md") {
            Write-Host "Analysis report: ANALISE_DESEMPENHO.md" -ForegroundColor Green
        }
    } else {
        Write-Host "generate_performance_graphs.py not found" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Magenta
Write-Host "                        Complete!                                   " -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Check metrics: results/metrics_cuda.csv, results/metrics_cuda.txt" -ForegroundColor Cyan
Write-Host "  2. Check validation: results/validation_cuda.txt" -ForegroundColor Cyan
Write-Host "  3. Check block size test: results/block_size_test.csv" -ForegroundColor Cyan
if ($python_exists) {
    Write-Host "  4. Review graphs in: graphs/" -ForegroundColor Cyan
    Write-Host "  5. Read analysis: ANALISE_DESEMPENHO.md" -ForegroundColor Cyan
}
Write-Host ""

Pop-Location
