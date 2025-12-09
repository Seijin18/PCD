# CUDA architecture (default: sm_60)
CUDA_ARCH ?= sm_75
# Makefile - K-Means 1D PCD (Windows + Linux Compatible)
# Compilação e testes conforme enunciado do projeto
# Compatível com Windows (cmd.exe) e Linux (bash)

# Detectar SO
ifeq ($(OS),Windows_NT)
    # Windows
    RM = del /Q
    RMDIR = rmdir /S /Q
    MKDIR = if not exist
    SHELL = cmd.exe
    SEP = \\
    EXE_EXT = .exe
else
    # Linux/Mac
    RM = rm -f
    RMDIR = rm -rf
    MKDIR = mkdir -p
    SEP = /
    EXE_EXT =
endif

CC = gcc
NVCC = nvcc
CFLAGS = -O2 -std=c99 -Wall -Wextra -lm
OPENMP_FLAGS = -O2 -std=c99 -fopenmp -Wall -Wextra -lm
CUDA_FLAGS = -O3
MPICC = mpicc

# Diretórios
BINDIR = bin
DATADIR = Data
RESULTSDIR = results

# Executáveis
SERIAL = $(BINDIR)/kmeans_serial$(EXE_EXT)
OMP = $(BINDIR)/kmeans_omp$(EXE_EXT)
CUDA = $(BINDIR)/kmeans_cuda$(EXE_EXT)
MPI = $(BINDIR)/kmeans_mpi$(EXE_EXT)
GEN_DATA = $(BINDIR)/gen_data$(EXE_EXT)

# Padrões de teste conforme enunciado
SMALL_N = 10000
SMALL_K = 4
MEDIUM_N = 100000
MEDIUM_K = 8
LARGE_N = 1000000
LARGE_K = 16

SEED = 42
MAX_ITER = 50
EPS = 1e-4

# Threads para OpenMP
OMP_THREADS = 1 2 4 8 16

.PHONY: all clean clean-results help serial omp cuda generate test test-serial test-omp test-cuda benchmark

all: $(SERIAL) $(OMP) $(CUDA) $(GEN_DATA)
	@echo ""
	@echo ======================================
	@echo COMPILACAO CONCLUIDA COM SUCESSO
	@echo ======================================
	@echo ""
	@echo Executaveis criados:
	@echo   - $(SERIAL)
	@echo   - $(OMP)
	@echo   - $(CUDA)
	@echo   - $(GEN_DATA)
	@echo ""
	@echo Use "make help" para ver opcoes disponiveis
	@echo ""

# ===== CRIAÇÃO DE DIRETÓRIOS =====

$(BINDIR):
	@echo Criando diretorio $(BINDIR)...
ifeq ($(OS),Windows_NT)
	@if not exist $(BINDIR) mkdir $(BINDIR)
else
	@mkdir -p $(BINDIR)
endif

$(DATADIR):
	@echo Criando diretorio $(DATADIR)...
ifeq ($(OS),Windows_NT)
	@if not exist $(DATADIR) mkdir $(DATADIR)
else
	@mkdir -p $(DATADIR)
endif

$(RESULTSDIR):
	@echo Criando diretorio $(RESULTSDIR)...
ifeq ($(OS),Windows_NT)
	@if not exist $(RESULTSDIR) mkdir $(RESULTSDIR)
else
	@mkdir -p $(RESULTSDIR)
endif

# ===== COMPILAÇÃO =====

$(SERIAL): Tools/kmeans_1d_serial.c | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $<
	@echo [OK] Serial compilado

$(OMP): OpenMP/kmeans_1d_omp.c | $(BINDIR)
	$(CC) $(OPENMP_FLAGS) -o $@ $<
	@echo [OK] OpenMP compilado

$(CUDA): Cuda/kmeans_1d_cuda.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) -arch=$(CUDA_ARCH) -o $@ $<
	@echo [OK] CUDA compilado

$(MPI): kmeans_1d_mpi.c | $(BINDIR)
	$(MPICC) -O2 -std=c99 -o $@ $< -lm
	@echo [OK] MPI compilado

$(GEN_DATA): Tools/generate_data.c | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $<
	@echo [OK] Gerador compilado

serial: $(SERIAL)
omp: $(OMP)
cuda: $(CUDA)

# ===== GERAÇÃO DE DADOS =====

generate: generate-small generate-medium
	@echo ""
	@echo Datasets principais gerados!
	@echo ""

generate-small: $(GEN_DATA) $(DATADIR)
	@echo Gerando dataset pequeno (N=$(SMALL_N), K=$(SMALL_K))...
	$(GEN_DATA) $(DATADIR)/dados_small.csv $(DATADIR)/centroides_small.csv $(SMALL_N) $(SMALL_K) $(SEED)
	@echo [OK] Dataset pequeno gerado

generate-medium: $(GEN_DATA) $(DATADIR)
	@echo Gerando dataset medio (N=$(MEDIUM_N), K=$(MEDIUM_K))...
	$(GEN_DATA) $(DATADIR)/dados_medium.csv $(DATADIR)/centroides_medium.csv $(MEDIUM_N) $(MEDIUM_K) $(SEED)
	@echo [OK] Dataset medio gerado

generate-large: $(GEN_DATA) $(DATADIR)
	@echo Gerando dataset grande (N=$(LARGE_N), K=$(LARGE_K))...
	@echo ATENCAO: Isso pode levar varios minutos...
	$(GEN_DATA) $(DATADIR)/dados_large.csv $(DATADIR)/centroides_large.csv $(LARGE_N) $(LARGE_K) $(SEED)
	@echo [OK] Dataset grande gerado

# ===== TESTES RÁPIDOS =====

test: test-serial test-omp
	@echo ""
	@echo ======================================
	@echo TODOS OS TESTES CONCLUIDOS
	@echo ======================================
	@echo ""

test-serial: $(SERIAL) generate-small $(RESULTSDIR)
	@echo ""
	@echo ======================================
	@echo TESTE: SERIAL (Pequeno)
	@echo ======================================
	@echo ""
	$(SERIAL) $(DATADIR)/dados_small.csv $(DATADIR)/centroides_small.csv $(SMALL_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_serial_test.csv $(RESULTSDIR)/centroids_serial_test.csv
	@echo ""

test-omp: $(OMP) generate-small $(RESULTSDIR)
	@echo ""
	@echo ======================================
	@echo TESTE: OPENMP (Pequeno, 4 threads)
	@echo ======================================
	@echo ""
	set OMP_NUM_THREADS=4
	$(OMP) $(DATADIR)/dados_small.csv $(DATADIR)/centroides_small.csv $(SMALL_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_omp_test.csv $(RESULTSDIR)/centroids_omp_test.csv
	@echo ""

test-cuda: $(CUDA) generate-small $(RESULTSDIR)
	@echo ""
	@echo ======================================
	@echo TESTE: CUDA (Pequeno)
	@echo ======================================
	@echo ""
	$(CUDA) $(DATADIR)/dados_small.csv $(DATADIR)/centroides_small.csv $(SMALL_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_cuda_test.csv $(RESULTSDIR)/centroids_cuda_test.csv
	@echo ""

# ===== TESTES COM DATASET GRANDE (LOG OUTPUT) =====

test-large: $(SERIAL) $(OMP) $(CUDA) $(RESULTSDIR)
	@echo ""
	@echo ======================================
	@echo TESTE: SERIAL (Large)
	@echo ======================================
	@echo ""
	$(SERIAL) $(DATADIR)/dados_large.csv $(DATADIR)/centroides_large.csv $(LARGE_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_serial_large.csv $(RESULTSDIR)/centroids_serial_large.csv > $(RESULTSDIR)/serial_large_run.txt 2>&1
	@echo ""
	@echo ======================================
	@echo TESTE: OPENMP (Large, var threads)
	@echo ======================================
	@echo ""
	@for %%T in (1 2 4 8 16) do (echo Threads: %%T & set OMP_NUM_THREADS=%%T & $(OMP) $(DATADIR)/dados_large.csv $(DATADIR)/centroides_large.csv $(LARGE_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_omp_%%T_large.csv $(RESULTSDIR)/centroids_omp_%%T_large.csv) > $(RESULTSDIR)/omp_large_runs.txt 2>&1
	@echo ""
	@echo ======================================
	@echo TESTE: CUDA (Large)
	@echo ======================================
	@echo ""
	$(CUDA) $(DATADIR)/dados_large.csv $(DATADIR)/centroides_large.csv $(LARGE_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_cuda_large.csv $(RESULTSDIR)/centroids_cuda_large.csv > $(RESULTSDIR)/cuda_large_run.txt 2>&1
	@echo ""
	@echo "Large dataset tests complete. Logs in $(RESULTSDIR)/"

# ===== BENCHMARK SERIAL =====

benchmark-serial: $(SERIAL) generate-small generate-medium $(RESULTSDIR)
	@echo ""
	@echo ======================================
	@echo BENCHMARK: SERIAL (Baseline)
	@echo ======================================
	@echo ""
	@echo --- Pequeno (N=$(SMALL_N), K=$(SMALL_K)) ---
	$(SERIAL) $(DATADIR)/dados_small.csv $(DATADIR)/centroides_small.csv $(SMALL_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_serial_small.csv $(RESULTSDIR)/centroids_serial_small.csv
	@echo ""
	@echo --- Medio (N=$(MEDIUM_N), K=$(MEDIUM_K)) ---
	$(SERIAL) $(DATADIR)/dados_medium.csv $(DATADIR)/centroides_medium.csv $(MEDIUM_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_serial_medium.csv $(RESULTSDIR)/centroids_serial_medium.csv
	@echo ""

# ===== BENCHMARK OPENMP =====

benchmark-omp: $(OMP) generate-small generate-medium $(RESULTSDIR)
	@echo ""
	@echo ======================================
	@echo BENCHMARK: OPENMP
	@echo ======================================
	@echo ""
	@echo --- PEQUENO (N=$(SMALL_N), K=$(SMALL_K)) ---
	@for %%T in (1 2 4 8 16) do (echo "" & echo Threads: %%T & set OMP_NUM_THREADS=%%T & $(OMP) $(DATADIR)/dados_small.csv $(DATADIR)/centroides_small.csv $(SMALL_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_omp_%%T_small.csv $(RESULTSDIR)/centroids_omp_%%T_small.csv)
	@echo ""
	@echo --- MEDIO (N=$(MEDIUM_N), K=$(MEDIUM_K)) ---
	@for %%T in (1 2 4 8 16) do (echo "" & echo Threads: %%T & set OMP_NUM_THREADS=%%T & $(OMP) $(DATADIR)/dados_medium.csv $(DATADIR)/centroides_medium.csv $(MEDIUM_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_omp_%%T_medium.csv $(RESULTSDIR)/centroids_omp_%%T_medium.csv)
	@echo ""

# ===== BENCHMARK CUDA =====

benchmark-cuda: $(CUDA) generate-small generate-medium $(RESULTSDIR)
	@echo ""
	@echo ======================================
	@echo BENCHMARK: CUDA
	@echo ======================================
	@echo ""
	@echo --- Pequeno (N=$(SMALL_N), K=$(SMALL_K)) ---
	$(CUDA) $(DATADIR)/dados_small.csv $(DATADIR)/centroides_small.csv $(SMALL_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_cuda_small.csv $(RESULTSDIR)/centroids_cuda_small.csv
	@echo ""
	@echo --- Medio (N=$(MEDIUM_N), K=$(MEDIUM_K)) ---
	$(CUDA) $(DATADIR)/dados_medium.csv $(DATADIR)/centroides_medium.csv $(MEDIUM_K) $(MAX_ITER) $(EPS) $(RESULTSDIR)/assign_cuda_medium.csv $(RESULTSDIR)/centroids_cuda_medium.csv
	@echo ""

# ===== BENCHMARK COMPLETO =====

benchmark: benchmark-serial benchmark-omp benchmark-cuda
	@echo ""
	@echo ======================================
	@echo BENCHMARK COMPLETO CONCLUIDO
	@echo ======================================
	@echo Resultados salvos em $(RESULTSDIR)/
	@echo ""

# ===== LIMPEZA =====

clean:
	@echo Limpando arquivos compilados...
ifeq ($(OS),Windows_NT)
	@if exist $(BINDIR) $(RMDIR) $(BINDIR)
	@if exist $(RESULTSDIR) $(RMDIR) $(RESULTSDIR)
else
	@$(RMDIR) $(BINDIR) 2>/dev/null || true
	@$(RMDIR) $(RESULTSDIR) 2>/dev/null || true
endif
	@echo [OK] Limpeza concluida

# Limpa apenas os resultados (preserva binarios e dados)
clean-results:
	@echo Limpando apenas a pasta de resultados ($(RESULTSDIR))...
ifeq ($(OS),Windows_NT)
	@if exist $(RESULTSDIR) $(RMDIR) $(RESULTSDIR)
	@if not exist $(RESULTSDIR) mkdir $(RESULTSDIR)
else
	@$(RMDIR) $(RESULTSDIR) 2>/dev/null || true
	@mkdir -p $(RESULTSDIR) 2>/dev/null || true
endif
	@echo [OK] Resultados limpos

distclean: clean
	@echo Removendo dados gerados...
ifeq ($(OS),Windows_NT)
	@if exist $(DATADIR) $(RMDIR) $(DATADIR)
else
	@$(RMDIR) $(DATADIR) 2>/dev/null || true
endif
	@echo [OK] Tudo removido

# ===== AJUDA =====

help:
	@echo.
	@echo ======================================
	@echo K-Means 1D PCD - Makefile
	@echo ======================================
	@echo.
	@echo COMPILACAO:
	@echo   make all        - Compila tudo (serial, omp, cuda, gen_data)
	@echo   make serial     - Compila apenas serial
	@echo   make omp        - Compila apenas OpenMP
	@echo   make cuda       - Compila apenas CUDA
	@echo.
	@echo GERACAO DE DADOS:
	@echo   make generate         - Gera datasets pequeno e medio
	@echo   make generate-small   - Dataset pequeno (N=10k, K=4)
	@echo   make generate-medium  - Dataset medio (N=100k, K=8)
	@echo   make generate-large   - Dataset grande (N=1M, K=16)
	@echo.
	@echo TESTES RAPIDOS:
	@echo   make test             - Testa serial e openmp (rapido)
	@echo   make test-serial      - Testa serial
	@echo   make test-omp         - Testa OpenMP (4 threads)
	@echo   make test-cuda        - Testa CUDA
	@echo.
	@echo BENCHMARK:
	@echo   make benchmark              - Executa todos os benchmarks
	@echo   make benchmark-serial       - Serial em 2 tamanhos
	@echo   make benchmark-omp          - OpenMP (todos os tamanhos x 5 threads)
	@echo   make benchmark-cuda         - CUDA (todos os tamanhos)
	@echo.
	@echo LIMPEZA:
	@echo   make clean            - Remove binarios e resultados
	@echo   make clean-results    - Remove apenas arquivos de resultados (preserva bin/ e Data/)
	@echo   make distclean        - Remove tudo (inclui dados)
	@echo.
	@echo PARAMETROS DE TESTE:
	@echo   Pequeno:  N=$(SMALL_N), K=$(SMALL_K)
	@echo   Medio:    N=$(MEDIUM_N), K=$(MEDIUM_K)
	@echo   Grande:   N=$(LARGE_N), K=$(LARGE_K)
	@echo   Threads OpenMP: $(OMP_THREADS)
	@echo   max_iter: $(MAX_ITER), eps: $(EPS), seed: $(SEED)
	@echo.
	@echo EXEMPLOS:
	@echo   make all                    # Compila tudo
	@echo   make generate               # Gera datasets
	@echo   make test                   # Teste rapido
	@echo   make benchmark-serial       # Benchmark serial
	@echo.
