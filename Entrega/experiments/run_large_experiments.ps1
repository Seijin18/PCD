# Run large experiments: compile and run serial, OpenMP and CUDA on the large dataset
# Saves terminal output to results/*.txt for later analysis and plotting

param(
    [int]$max_iter = 50,
    [string]$eps = "1e-4",
    [int]$repetitions = 3,
    [int[]]$threads = @(1,2,4,8,16)
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Project root is parent of the experiments folder
$root = Join-Path $scriptDir ".."
Set-Location $root

# Ensure results directory exists
if (-not (Test-Path -Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

Write-Host "Compiling all targets (make all)..."
# Try using make, fallback to build.bat
$makeOK = $false
try {
    & make all
    if ($LASTEXITCODE -eq 0) { $makeOK = $true }
} catch {
    $makeOK = $false
}

if (-not $makeOK) {
    Write-Host "make failed or not present; attempting build.bat"
    cmd /c "build.bat build-seq" | Tee-Object -FilePath results\build_seq_log.txt
    cmd /c "build.bat build-omp" | Tee-Object -FilePath results\build_omp_log.txt
    cmd /c "build.bat build-cuda" | Tee-Object -FilePath results\build_cuda_log.txt
}

# Run serial (multiple repetitions)
$serialExe = "bin\\kmeans_serial.exe"
if (-not (Test-Path $serialExe)) { Write-Error "Serial executable not found: $serialExe"; exit 1 }
for ($r=1; $r -le $repetitions; $r++) {
    Write-Host "Running serial on large dataset (rep $r/$repetitions)..."
    $log = "results\\serial_large_run_rep${r}.txt"
    $assign = "results\\assign_serial_large_rep${r}.csv"
    $cent = "results\\centroids_serial_large_rep${r}.csv"
    & $serialExe "Data\\dados_large.csv" "Data\\centroides_large.csv" 16 $max_iter $eps $assign $cent 2>&1 | Tee-Object -FilePath $log
}

# Run OpenMP for different thread counts (multiple repetitions)
$ompExe = "bin\\kmeans_omp.exe"
if (-not (Test-Path $ompExe)) { Write-Error "OpenMP executable not found: $ompExe"; exit 1 }
foreach ($t in $threads) {
    for ($r=1; $r -le $repetitions; $r++) {
        Write-Host "Running OpenMP with $t threads (rep $r/$repetitions)..."
        $env:OMP_NUM_THREADS = $t
        $log = "results\\omp_${t}_large_run_rep${r}.txt"
        $assign = "results\\assign_omp_${t}_large_rep${r}.csv"
        $cent = "results\\centroids_omp_${t}_large_rep${r}.csv"
        & $ompExe "Data\\dados_large.csv" "Data\\centroides_large.csv" 16 $max_iter $eps $assign $cent 2>&1 | Tee-Object -FilePath $log
    }
}

# Run CUDA (multiple repetitions)
$cudaExe = "bin\\kmeans_cuda.exe"
if (Test-Path $cudaExe) {
    for ($r=1; $r -le $repetitions; $r++) {
        Write-Host "Running CUDA on large dataset (rep $r/$repetitions)..."
        $log = "results\\cuda_large_run_rep${r}.txt"
        $assign = "results\\assign_cuda_large_rep${r}.csv"
        $cent = "results\\centroids_cuda_large_rep${r}.csv"
        & $cudaExe "Data\\dados_large.csv" "Data\\centroides_large.csv" 16 $max_iter $eps $assign $cent 2>&1 | Tee-Object -FilePath $log
    }
} else {
    Write-Host "CUDA executable not found (bin\\kmeans_cuda.exe). Skipping CUDA run." | Tee-Object -FilePath results\\cuda_large_run.txt
}

Write-Host "All runs complete. Logs saved in results\\"
