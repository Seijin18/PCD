@echo off
REM ============================================================
REM K-Means 1D CUDA - Build Script (Windows)
REM ============================================================
REM 
REM Usage:
REM   build.bat              - Build completo
REM   build.bat data         - Gera dados
REM   build.bat data-large   - Gera dados grandes
REM   build.bat build-cuda   - Compila CUDA
REM   build.bat run-cuda     - Executa CUDA
REM   build.bat compare      - Compila e executa ambas
REM   build.bat clean        - Limpa arquivos
REM   build.bat help         - Mostra esta mensagem

setlocal enabledelayedexpansion

rem Directory of this script (always ends with a backslash)
set SCRIPT_DIR=%~dp0

set TOOLS_DIR=%SCRIPT_DIR%Tools
set CUDA_DIR=%SCRIPT_DIR%Cuda
set OMP_DIR=%SCRIPT_DIR%OpenMP
set DATA_DIR=%SCRIPT_DIR%Data
set RESULTS_DIR=%SCRIPT_DIR%results
set BIN_DIR=%SCRIPT_DIR%bin

set N=100000
set K=20
set SEED=42
set MAX_ITER=50
set EPS=1e-4
set CUDA_ARCH=sm_75
set SMALL_N=10000
set SMALL_K=4
set MEDIUM_N=100000
set MEDIUM_K=8
set LARGE_N=1000000
set LARGE_K=16

REM Processar argumentos
if "%1"=="" goto build
if /i "%1"=="help" goto help
if /i "%1"=="data" goto gen_data
if /i "%1"=="data-large" goto gen_data_large
if /i "%1"=="build-cuda" goto build_cuda
if /i "%1"=="build-seq" goto build_seq
if /i "%1"=="build-omp" goto build_omp
if /i "%1"=="run-cuda" goto run_cuda
if /i "%1"=="run-seq" goto run_seq
if /i "%1"=="run-omp" goto run_omp
if /i "%1"=="compare" goto compare
if /i "%1"=="compare-all" goto compare_all
if /i "%1"=="clean" goto clean
if /i "%1"=="clean-all" goto clean_all

REM Default: build completo
:build
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         K-Means 1D CUDA - Build Completo                     ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

if not exist "%DATA_DIR%\dados.csv" (
    echo Dados nao encontrados. Gerando...
    call :gen_data
)

if not exist "%BIN_DIR%" (
    mkdir "%BIN_DIR%"
)

call :build_seq
call :build_omp
call :build_cuda
echo.
echo ✓ Build completo finalizado!
echo   Para executar: build.bat run-cuda
goto end

:help
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         K-Means 1D CUDA - Build Script (Windows)              ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo TARGETS:
echo   build.bat              - Build completo
echo   build.bat data         - Gera dados (N=100K, K=20)
echo   build.bat data-large   - Gera dados grandes (N=1M, K=20)
echo   build.bat build-cuda   - Compila CUDA
echo   build.bat build-seq    - Compila sequencial
echo   build.bat build-omp    - Compila OpenMP
echo   build.bat run-cuda     - Executa CUDA
echo   build.bat run-seq      - Executa sequencial
echo   build.bat run-omp      - Executa OpenMP
echo   build.bat compare      - Compila e executa seq + CUDA
echo   build.bat compare-all  - Compila e executa seq + OpenMP + CUDA
echo   build.bat clean        - Remove executaveis e resultados
echo   build.bat clean-all    - Remove tudo (inclui dados)
echo.
goto end

:gen_data
setlocal
if "%2" neq "" set N=%2
if "%3" neq "" set K=%3

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Gerando Dados para K-Means                            ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Parametros:
echo   N (pontos):  %N%
echo   K (clusters): %K%
echo   Seed:        %SEED%
echo.

if not exist "%BIN_DIR%\gen_data.exe" (
    echo Compilando gerador de dados...
    gcc -O3 -std=c99 "%TOOLS_DIR%\generate_data.c" -o "%BIN_DIR%\gen_data.exe" -lm
    if errorlevel 1 (
        echo Erro na compilacao!
        goto end
    )
)

rem Generate small, medium, large datasets used by Makefile targets
"%BIN_DIR%\gen_data.exe" "%DATA_DIR%\dados_small.csv" "%DATA_DIR%\centroides_small.csv" %SMALL_N% %SMALL_K% %SEED%
if errorlevel 1 (
    echo Erro ao gerar dados pequenos!
    goto end
)

"%BIN_DIR%\gen_data.exe" "%DATA_DIR%\dados_medium.csv" "%DATA_DIR%\centroides_medium.csv" %MEDIUM_N% %MEDIUM_K% %SEED%
if errorlevel 1 (
    echo Erro ao gerar dados medios!
    goto end
)

"%BIN_DIR%\gen_data.exe" "%DATA_DIR%\dados_large.csv" "%DATA_DIR%\centroides_large.csv" %LARGE_N% %LARGE_K% %SEED%
if errorlevel 1 (
    echo Erro ao gerar dados grandes!
    goto end
)

rem Also produce legacy filenames for compatibility
"%BIN_DIR%\gen_data.exe" "%DATA_DIR%\dados.csv" "%DATA_DIR%\centroides_iniciais.csv" %N% %K% %SEED%
if errorlevel 1 (
    echo Erro ao gerar dados (legacy)!
    goto end
)

echo.
echo ✓ Dados gerados em %DATA_DIR%\
endlocal
goto end

:gen_data_large
setlocal
set N=1000000
set K=20
echo Gerando dataset grande...
call :gen_data
endlocal
goto end

:build_cuda
setlocal
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Compilando K-Means CUDA                               ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Detectar compute capability da GPU
echo Detectando GPU...
for /f "delims=" %%a in ('nvidia-smi --query-gpu=name --format=csv,noheader') do (
    echo GPU: %%a
)

REM Compilar
echo Compilando com NVCC...
if exist "%CUDA_DIR%\kmeans_1d_cuda.cu" (
    nvcc -O3 -arch=%CUDA_ARCH% "%CUDA_DIR%\kmeans_1d_cuda.cu" -o "%BIN_DIR%\kmeans_cuda.exe"
    if errorlevel 1 (
        echo Erro na compilacao CUDA!
        goto end
    )
    echo ✓ Compilado: %BIN_DIR%\kmeans_cuda.exe
) else (
    echo Arquivo %CUDA_DIR%\\kmeans_1d_cuda.cu nao encontrado. Pulando compilacao CUDA.
)
endlocal
goto end

:build_seq
setlocal
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Compilando K-Means Sequencial                         ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

gcc -O3 -std=c99 "%TOOLS_DIR%\kmeans_1d_serial.c" -o "%BIN_DIR%\kmeans_serial.exe" -lm
if errorlevel 1 (
    echo Erro na compilacao!
    goto end
)

echo ✓ Compilado: %BIN_DIR%\kmeans_serial.exe
endlocal
goto end

:build_omp
setlocal
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Compilando K-Means OpenMP                             ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

gcc -O3 -std=c99 -fopenmp "%OMP_DIR%\kmeans_1d_omp.c" -o "%BIN_DIR%\kmeans_omp.exe" -lm
if errorlevel 1 (
    echo Erro na compilacao!
    goto end
)

echo ✓ Compilado: %BIN_DIR%\kmeans_omp.exe
endlocal

:run_cuda
setlocal
if not exist "%DATA_DIR%\dados_small.csv" (
    echo Dados nao encontrados! Gerando...
    call :gen_data
)

if not exist "%BIN_DIR%\kmeans_cuda.exe" (
    echo Compilavel CUDA nao encontrado. Compilando...
    call :build_cuda
)
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Executando K-Means CUDA                               ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Parametros:
echo   K:       %K%
echo   Maxiter: %MAX_ITER%
echo   Epsilon: %EPS%
echo.

"%BIN_DIR%\kmeans_cuda.exe" "%DATA_DIR%\dados_small.csv" "%DATA_DIR%\centroides_small.csv" %K% %MAX_ITER% %EPS%

endlocal
goto end

:run_omp
setlocal
if not exist "%BIN_DIR%\kmeans_omp.exe" (
    echo Compilando versao OpenMP...
    call :build_omp
)

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Executando K-Means OpenMP                             ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Parametros:
echo   K:       %K%
echo   Maxiter: %MAX_ITER%
echo   Epsilon: %EPS%
echo.

"%BIN_DIR%\kmeans_omp.exe" "%DATA_DIR%\dados_small.csv" "%DATA_DIR%\centroides_small.csv" %K% %MAX_ITER% %EPS%

endlocal
goto end

:compare
setlocal
if not exist "%BIN_DIR%\kmeans_serial.exe" (
    echo Compilando versao sequencial...
    call :build_seq
)

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Executando K-Means Sequencial                         ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Parametros:
echo   K:       %K%
echo   Maxiter: %MAX_ITER%
echo   Epsilon: %EPS%
echo.

"%BIN_DIR%\kmeans_serial.exe" "%DATA_DIR%\dados_small.csv" "%DATA_DIR%\centroides_small.csv" %K% %MAX_ITER% %EPS%

endlocal
goto end

:compare
call :run_seq
call :run_omp
call :run_cuda
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Comparacao de Resultados (Seq vs OpenMP vs CUDA)     ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Execução de todas as versões concluída!
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Resultados salvos em %RESULTS_DIR%\\
echo Use 'build.bat graphs' para gerar graficos (requer Python)
echo.
goto end
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         Comparacao de Resultados (Seq vs OpenMP vs CUDA)     ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Execução de todas as versões concluída!
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Resultados salvos em %CUDA_DIR%\results\
echo Use 'build.bat graphs' para gerar graficos (requer Python)
echo.
goto end

:clean
echo.
echo Limpando executaveis e resultados...
del /q "%CUDA_DIR%\*.exe" 2>nul
del /q "%BIN_DIR%\*.exe" 2>nul
del /q "%RESULTS_DIR%\*" 2>nul
del /q "%TOOLS_DIR%\gen_data.exe" 2>nul
if exist "%CUDA_DIR%\results" rmdir /q "%CUDA_DIR%\results"
if exist "%CUDA_DIR%\graphs" rmdir /q "%CUDA_DIR%\graphs"
if exist "%RESULTS_DIR%" rmdir /q "%RESULTS_DIR%"
if exist "%BIN_DIR%" rmdir /q "%BIN_DIR%"
echo ✓ Limpeza concluida
goto end

:clean_all
call :clean
echo Removendo dados gerados...
del /q "%DATA_DIR%\*.csv" 2>nul
echo ✓ Dados removidos
goto end

:end
endlocal
exit /b 0
