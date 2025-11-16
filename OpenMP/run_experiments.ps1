# Script PowerShell para executar experimentos K-Means
# Compara versão serial com versão paralela OpenMP

Write-Host "=== Experimentos K-Means 1D ===" -ForegroundColor Cyan
Write-Host ""

# Verificar se GCC está disponível
$gcc = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $gcc) {
    Write-Host "ERRO: GCC não encontrado. Instale o MinGW ou similar." -ForegroundColor Red
    exit 1
}

# Gerar dados de teste
Write-Host "1. Gerando dados de teste..." -ForegroundColor Yellow
if (Test-Path "dados.csv") {
    Write-Host "   Usando dados existentes (dados.csv)" -ForegroundColor Gray
} else {
    & D:/Projetinhos/Faculdade/.venv/Scripts/python.exe generate_data.py 10000 3 42
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRO ao gerar dados!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Compilar versão serial
Write-Host "2. Compilando versão serial..." -ForegroundColor Yellow
gcc -O2 -std=c99 kmeans_1d_serial.c -o kmeans_1d_serial.exe -lm
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO na compilação da versão serial!" -ForegroundColor Red
    exit 1
}
Write-Host "   Compilado: kmeans_1d_serial.exe" -ForegroundColor Green
Write-Host ""

# Compilar versão OpenMP
Write-Host "3. Compilando versão OpenMP..." -ForegroundColor Yellow
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp.exe -lm
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRO na compilação da versão OpenMP!" -ForegroundColor Red
    exit 1
}
Write-Host "   Compilado: kmeans_1d_omp.exe" -ForegroundColor Green
Write-Host ""

# Executar versão serial
Write-Host "4. Executando versão SERIAL..." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan

$serial_times = @()
$num_runs = 5

for ($i = 1; $i -le $num_runs; $i++) {
    Write-Host "   Execução $i/$num_runs" -ForegroundColor Gray
    $output = .\kmeans_1d_serial.exe dados.csv centroides_iniciais.csv 20
    
    # Extrair tempo da saída
    $time_line = $output | Select-String "Tempo total: ([\d.]+) ms"
    if ($time_line) {
        $time = [double]($time_line.Matches.Groups[1].Value)
        $serial_times += $time
        Write-Host "      Tempo: $time ms" -ForegroundColor Gray
    }
}

$serial_avg = ($serial_times | Measure-Object -Average).Average
Write-Host ""
Write-Host "   Tempo médio (serial): $([math]::Round($serial_avg, 3)) ms" -ForegroundColor Green
Write-Host ""

# Executar versão paralela com diferentes números de threads
Write-Host "5. Executando versão PARALELA (OpenMP)..." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan

$thread_counts = @(1, 2, 4, 8, 16)
$results = @()

foreach ($threads in $thread_counts) {
    Write-Host ""
    Write-Host "   Testando com $threads thread(s)..." -ForegroundColor Magenta
    
    $parallel_times = @()
    
    for ($i = 1; $i -le $num_runs; $i++) {
        Write-Host "      Execução $i/$num_runs" -ForegroundColor Gray
        $output = .\kmeans_1d_omp.exe dados.csv centroides_iniciais.csv 20 $threads
        
        # Extrair tempo da saída
        $time_line = $output | Select-String "Tempo total: ([\d.]+) ms"
        if ($time_line) {
            $time = [double]($time_line.Matches.Groups[1].Value)
            $parallel_times += $time
            Write-Host "         Tempo: $time ms" -ForegroundColor Gray
        }
    }
    
    $parallel_avg = ($parallel_times | Measure-Object -Average).Average
    $speedup = $serial_avg / $parallel_avg
    
    Write-Host "      Tempo médio: $([math]::Round($parallel_avg, 3)) ms" -ForegroundColor Green
    Write-Host "      Speedup: $([math]::Round($speedup, 3))x" -ForegroundColor Cyan
    
    $results += [PSCustomObject]@{
        Threads = $threads
        AvgTime = [math]::Round($parallel_avg, 3)
        Speedup = [math]::Round($speedup, 3)
        Efficiency = [math]::Round(($speedup / $threads) * 100, 1)
    }
}

# Mostrar resumo
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "6. RESUMO DOS RESULTADOS" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tempo médio serial: $([math]::Round($serial_avg, 3)) ms" -ForegroundColor White
Write-Host ""
Write-Host "Resultados paralelos:" -ForegroundColor White
$results | Format-Table -AutoSize

# Verificar corretude comparando SSE
Write-Host ""
Write-Host "7. Verificando corretude (primeiros 1000 pontos)..." -ForegroundColor Yellow

# Comparar primeiros 1000 pontos (para economizar tempo)
$serial_assign = (Get-Content "assign_serial.csv" -TotalCount 1000)
$issues = 0

foreach ($threads in $thread_counts) {
    $parallel_assign = (Get-Content "assign_omp_$threads.csv" -TotalCount 1000)
    
    if (Compare-Object $serial_assign $parallel_assign) {
        Write-Host "   AVISO: Atribuições diferentes para $threads threads!" -ForegroundColor Yellow
        $issues++
    } else {
        Write-Host "   OK: Atribuições idênticas para $threads threads" -ForegroundColor Green
    }
}

if ($issues -eq 0) {
    Write-Host ""
    Write-Host "Todas as verificações passaram!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "ATENÇÃO: Algumas diferenças foram encontradas!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Experimentos concluídos!" -ForegroundColor Cyan
