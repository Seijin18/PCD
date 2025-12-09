# Projeto K-Means 1D - CUDA GPU Otimizado

Implementação otimizada do algoritmo K-Means 1D com paralelização em GPU usando CUDA.

## 🎯 Objetivo

Comparar o desempenho da implementação do K-Means 1D entre:
- **CPU (Sequencial):** Versão otimizada em C para linha de base
- **GPU (CUDA):** Versão paralelizada com otimizações avançadas

## 📊 Características

### Versão Sequencial (CPU)
- **Arquivo:** `kmeans_1d_seq.c`
- **Compilador:** GCC/MSVC
- **Otimizações:** -O3, cache-friendly allocation
- **Tempo de medição:** clock() de alta precisão

### Versão CUDA (GPU) - Otimizada
- **Arquivo:** `kmeans_1d_cuda_optimized.cu`
- **Compilador:** NVCC (NVIDIA CUDA Compiler)
- **Otimizações Implementadas:**
  - **Memória Constante:** Centróides em cache constante (acesso rápido)
  - **Redução por Blocos:** Agregação eficiente usando shared memory
  - **Block Size Automático:** Teste e seleção automática do tamanho ótimo
  - **SSE no Host:** Cálculo de SSE na CPU para reduzir overhead
- **Kernels:**
  - `kernel_assignment_optimized`: Atribuição com memória constante
  - `kernel_update_reduction`: Agregação eficiente por blocos
- **Tempo de medição:** cudaEvent com precisão de microssegundos

## 📁 Estrutura do Projeto

```
Cuda/
├── data/                              # 📥 Dados de entrada
│   ├── dados.csv                      # Dataset (100,000 pontos)
│   └── centroides_iniciais.csv        # Centróides iniciais (K=20)
│
├── results/                           # 📊 Resultados e métricas
│   ├── assign_cuda.csv               # Atribuições GPU
│   ├── assign_seq.csv                # Atribuições CPU
│   ├── centroids_cuda.csv            # Centróides finais GPU
│   ├── centroids_seq.csv             # Centróides finais CPU
│   ├── block_size_test.csv           # Teste de tamanhos de bloco
│   ├── metrics_cuda.csv              # Métricas estruturadas
│   ├── metrics_cuda.txt              # Métricas legíveis
│   ├── validation_cuda.txt           # Validação GPU vs CPU
│   └── comparacao_seq_vs_cuda.txt    # Comparação detalhada
│
├── graphs/                            # 📈 Gráficos (gerados)
│   ├── block_size_analysis.png
│   ├── throughput_analysis.png
│   ├── timing_breakdown.png
│   └── performance_summary.png
│
├── kmeans_1d_cuda_optimized.cu       # Implementação CUDA otimizada
├── kmeans_1d_seq.c                   # Implementação sequencial
├── generate_performance_graphs.py     # Geração de gráficos
├── generate_comparison.ps1            # Geração de relatório
├── build_and_analyze_cuda.ps1        # Build automático completo
└── README.md                          # Este arquivo
```

## 🚀 Como Usar

### Pré-requisitos

```powershell
# Windows
# - CUDA Toolkit 11.0+ (inclui NVCC)
# - Visual Studio 2019+ com C++ Build Tools
# - Python 3.8+ com pandas, matplotlib, numpy (opcional, para gráficos)

# Verificar CUDA e GPU
nvidia-smi

# Verificar NVCC
nvcc --version
```

### 1. Execução Automática Completa (Recomendado)

```powershell
cd Cuda
.\build_and_analyze_cuda.ps1
```

Este script realiza todo o workflow:
- ✓ Detecta automaticamente compute capability da GPU
- ✓ Compila versão CUDA otimizada
- ✓ Executa com teste automático de block sizes (32, 64, 128, 256, 512)
- ✓ Gera métricas de desempenho (CSV e TXT)
- ✓ Valida resultados contra versão sequencial
- ✓ Gera gráficos de análise (se Python disponível)
- ✓ Salva todos os resultados em `results/`

### 2. Compilação Manual

#### Versão Sequencial (CPU)
```bash
# Windows (MSVC)
cl /O2 kmeans_1d_seq.c /Fe:kmeans_1d_seq.exe

# Windows (GCC/MinGW)
gcc -O3 -std=c99 kmeans_1d_seq.c -o kmeans_1d_seq.exe
```

#### Versão CUDA (GPU)
```bash
# Detectar compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Compilar (exemplo para GTX 1660 Ti - sm_75)
nvcc -O3 -arch=sm_75 kmeans_1d_cuda_optimized.cu -o kmeans_1d_cuda_opt.exe
```

**Compute Capabilities comuns:**
- sm_75 = Turing (RTX 2060/2070/2080, GTX 1660/1660 Ti)
- sm_80 = Ampere (RTX 3060/3070/3080/3090)
- sm_86 = Ampere (RTX 3050, RTX 30 Mobile)
- sm_89 = Ada Lovelace (RTX 4060/4070)
- sm_90 = Ada Lovelace (RTX 4080/4090)

### 3. Executar Individualmente

#### Versão CPU
```powershell
.\kmeans_1d_seq.exe data/dados.csv data/centroides_iniciais.csv 20 100 1e-6
```

#### Versão GPU
```powershell
.\kmeans_1d_cuda_opt.exe data/dados.csv data/centroides_iniciais.csv 20 100 1e-6
```

**Parâmetros:**
- `data/dados.csv` - Arquivo de entrada com pontos
- `data/centroides_iniciais.csv` - Centróides iniciais
- `20` - Número de clusters (K)
- `100` - Número máximo de iterações
- `1e-6` - Epsilon de convergência

### 4. Gerar Gráficos de Desempenho

```powershell
python generate_performance_graphs.py .
```

Gera 4 gráficos profissionais:
- **block_size_analysis.png** - Análise de tamanhos de bloco
- **throughput_analysis.png** - Throughput e eficiência
- **timing_breakdown.png** - Decomposição de tempo
- **performance_summary.png** - Resumo geral (6 painéis)

### 5. Gerar Relatório de Comparação

```powershell
.\generate_comparison.ps1 -seq_time 208.0 -cuda_time 93.789
```

Cria `results/comparacao_seq_vs_cuda.txt` com análise detalhada.

## 📖 Algoritmo e Otimizações

### Assignment Step (GPU) - Com Memória Constante

```cuda
__constant__ double constant_centroids[MAX_K];

__global__ void kernel_assignment_optimized(double *data, int N, int K,
                                             int *assignments, double *sse_array) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    double point = data[i];
    double min_dist = 1e308;
    int best_cluster = 0;
    
    // Usar centróides da memória constante (cache rápido)
    for (int k = 0; k < K; k++) {
        double diff = point - constant_centroids[k];
        double dist = diff * diff;
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = k;
        }
    }
    
    assignments[i] = best_cluster;
    sse_array[i] = min_dist;
}
```

**Otimizações:**
- ✓ Memória constante para centróides (acesso em cache L1)
- ✓ Sem divergência de warp (todas as threads executam mesmo código)
- ✓ Complexidade: O(N × K) totalmente paralela

### Update Step (GPU) - Redução por Blocos

```cuda
__global__ void kernel_update_reduction(int *assignments, double *data, int N, int K,
                                         double *block_sums, int *block_counts) {
    extern __shared__ char shared_memory[];
    
    double *shared_sums = (double *)shared_memory;
    int *shared_counts = (int *)&shared_memory[K * sizeof(double)];
    
    // Inicializar shared memory
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        shared_sums[k] = 0.0;
        shared_counts[k] = 0;
    }
    __syncthreads();
    
    // Acumular em shared memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int cluster = assignments[i];
        atomicAdd(&shared_sums[cluster], data[i]);
        atomicAdd(&shared_counts[cluster], 1);
    }
    __syncthreads();
    
    // Escrever resultados do bloco para memória global
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        if (shared_sums[k] != 0.0 || shared_counts[k] != 0) {
            atomicAdd(&block_sums[k], shared_sums[k]);
            atomicAdd(&block_counts[k], shared_counts[k]);
        }
    }
}
```

**Otimizações:**
- ✓ Shared memory reduz acessos à memória global
- ✓ Operações atômicas apenas dentro do bloco (muito mais rápido)
- ✓ Reduz contenção de memória global significativamente

### Block Size Automático

O código testa automaticamente múltiplos tamanhos de bloco:
- **32, 64, 128, 256, 512 threads**
- Seleciona o melhor baseado em tempo de execução real
- Resultados salvos em `results/block_size_test.csv`

## 📊 Resultados de Desempenho

### Configuração Testada
- **Dataset:** 100,000 pontos
- **Clusters (K):** 20
- **Iterações:** 100
- **GPU:** NVIDIA GeForce GTX 1660 Ti (1536 CUDA cores, Compute 7.5)
- **CPU:** Intel/AMD (sequencial)
### Métricas de Desempenho

| Métrica | Sequencial (CPU) | CUDA (GPU) | Melhoria |
|---------|------------------|------------|----------|
| **Tempo Total** | 208.0 ms | 93.8 ms | **2.22x** |
| **Tempo/Iteração** | 2.08 ms | 0.938 ms | 2.22x |
| **Throughput** | 48.08 M pts/s | 107.15 M pts/s | 2.23x |
| **Overhead H2D** | - | 0.177 ms | 0.2% |
| **Tempo Kernels** | - | 93.329 ms | 99.5% |
| **Overhead D2H** | - | 0.283 ms | 0.3% |

### Validação de Corretude

| Verificação | Resultado |
|-------------|-----------|
| **Match Atribuições** | 100.00% (0 diferenças) |
| **Diferença SSE** | < 1e-10 (praticamente zero) |
| **Diferença Centróides** | 3.96e-11 (máxima) |
| **Status** | ✅ PASSOU |

### Block Size Ótimo

| Block Size | Tempo/Iteração |
|------------|----------------|
| 32 threads | 0.126 ms |
| **64 threads** | **0.111 ms ✓ MELHOR** |
| 128 threads | 0.111 ms |
| 256 threads | 0.114 ms |
| 512 threads | 0.123 ms |

**Configuração Ótima:**
- Block size: **64 threads**
- Grid size: 1563 blocos
- Ocupação: Ótima para Turing (sm_75)

## 📝 Arquivos de Saída

### Diretório `results/`

#### Métricas e Validação
- **metrics_cuda.csv** - Métricas estruturadas (CSV)
- **metrics_cuda.txt** - Métricas legíveis (texto)
- **block_size_test.csv** - Resultados de teste de block sizes
- **validation_cuda.txt** - Validação GPU vs CPU
- **comparacao_seq_vs_cuda.txt** - Comparação detalhada completa

#### Resultados do Algoritmo
- **assign_cuda.csv** / **assign_seq.csv** - Atribuições (N linhas)
- **centroids_cuda.csv** / **centroids_seq.csv** - Centróides finais (K linhas)

### Diretório `graphs/`

- **block_size_analysis.png** - Linha: tempo vs block size
- **throughput_analysis.png** - Barra + pizza: throughput e distribuição
- **timing_breakdown.png** - Barras: decomposição de tempo H2D/Kernels/D2H
- **performance_summary.png** - Dashboard 6 painéis: visão geral completa

## 🔧 Troubleshooting

### ERRO: "nvcc: command not found" ou "Cannot find compiler 'cl.exe'"

```powershell
# 1. Adicionar MSVC ao PATH (necessário no Windows)
$msvcPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
$env:PATH = "$msvcPath;" + $env:PATH

# 2. Adicionar CUDA ao PATH
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"

# 3. Verificar
nvcc --version
cl.exe
```

### ERRO: "Device does not support this compute capability"

```powershell
# Descobrir compute capability da GPU
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Usar o valor correto. Exemplo para GTX 1660 Ti (7.5):
nvcc -O3 -arch=sm_75 kmeans_1d_cuda_optimized.cu -o kmeans_1d_cuda_opt.exe
```

### GPU mais lenta que CPU

**Causas comuns:**
- Normal para N < 50K (overhead CUDA domina)
- Transferências PCI-E são gargalo em problemas pequenos
- **Solução:** Aumentar N para 500K-1M para ver speedup real

### Arquivos não encontrados

```powershell
# Verificar estrutura de diretórios
Get-ChildItem data/
Get-ChildItem results/

# Executar com caminhos corretos
.\kmeans_1d_cuda_opt.exe data/dados.csv data/centroides_iniciais.csv 20 100 1e-6
```

### Python não gera gráficos

```powershell
# Instalar dependências
pip install pandas matplotlib numpy

# Executar em ambiente virtual (recomendado)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas matplotlib numpy
python generate_performance_graphs.py .
```

## 💡 Dicas de Otimização

### Para Aumentar Speedup

1. **Aumentar tamanho do problema:**
   ```powershell
   # Gerar dataset maior (1M pontos)
   python generate_data.py 1000000 20
   ```

2. **Aumentar número de clusters (K):**
   - K maior = mais trabalho computacional
   - Melhor aproveitamento da GPU

3. **Usar problema multidimensional:**
   - K-Means 2D/3D tem muito mais operações
   - GPU se beneficia mais de problemas complexos

### Para Reduzir Overhead

- Minimizar transferências H2D/D2H
- Usar streams CUDA para sobreposição
- Pinned memory para transferências mais rápidas
- Processar múltiplos datasets em batch

## 📚 Referências

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Constant Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)
- [Shared Memory in CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory)
- [Atomic Operations in CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)
- [K-Means Clustering Algorithm](https://en.wikipedia.org/wiki/K-means_clustering)

## 📋 Notas Técnicas

### Implementação
- Versão CUDA usa `cudaEvent` para medição precisa de tempo na GPU
- Versão CPU usa `clock()` com detecção automática Windows/Linux
- Ambas garantem resultados determinísticos com mesma seed
- SSE calculado no host (CPU) para reduzir overhead de kernel

### Otimizações Aplicadas
1. **Memória Constante:** Cache L1 de 64KB, broadcast para threads do warp
2. **Shared Memory:** Redução de acessos à memória global (100x mais rápida)
3. **Block Size Ótimo:** Testado automaticamente para hardware específico
4. **Coalesced Memory Access:** Acesso sequencial otimizado aos dados

### Limitações
- Problema 1D tem baixa intensidade aritmética (poucos FLOPs por byte)
- Overhead de lançamento de kernel é significativo para N pequeno
- Speedup ideal requer N > 500K para saturar GPU moderna

## 🎯 Conclusões

### Desempenho Alcançado
- ✅ **Speedup de 2.22x** para 100K pontos
- ✅ **Overhead mínimo** de comunicação (0.5%)
- ✅ **100% de corretude** validada
- ✅ **Block size otimizado** automaticamente

### Recomendações
- Para **problemas pequenos** (N < 50K): CPU é mais eficiente
- Para **problemas médios** (50K < N < 500K): GPU oferece speedup moderado (2-3x)
- Para **problemas grandes** (N > 500K): GPU oferece speedup significativo (5-10x)
- Para **máximo desempenho**: Usar K-Means 2D/3D com mais operações por ponto

## 👨‍💻 Autor

Implementação para disciplina de Programação Concorrente e Distribuída

---

**Versão:** 2.0 (Otimizada)  
**Data:** Novembro 2025  
**GPU Testada:** NVIDIA GeForce GTX 1660 Ti (Compute 7.5)
