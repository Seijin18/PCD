# Projeto K-Means 1D - CUDA GPU

Implementação do algoritmo K-Means 1D com paralelização em GPU usando CUDA (Entrega 2).

## 🎯 Objetivo

Comparar o desempenho da implementação do K-Means 1D entre:
- **CPU (Sequencial):** Versão otimizada em C para linha de base
- **GPU (CUDA):** Versão paralelizada usando NVIDIA CUDA

## 📊 Características

### Versão Sequencial (CPU)
- **Arquivo:** `kmeans_1d_seq.c`
- **Compilador:** GCC/Clang
- **Otimizações:** -O3, cache-friendly allocation
- **Tempo de medição:** clock() de alta precisão

### Versão CUDA (GPU)
- **Arquivo:** `kmeans_1d_cuda.cu`
- **Compilador:** NVCC (NVIDIA CUDA Compiler)
- **Kernels:**
  - `kernel_assignment`: Atribuição paralela de pontos (1 thread por ponto)
  - `kernel_update_partial`: Acumulação paralela de somas (operações atômicas)
  - `kernel_update_centroids`: Cálculo paralelo de novos centróides
  - `kernel_reduce_sse`: Redução paralela do SSE em shared memory
- **Tempo de medição:** cudaEventElapsedTime() para precisão GPU

## 📁 Estrutura

```
Cuda/
├── kmeans_1d_seq.c              # Implementação sequencial (CPU)
├── kmeans_1d_cuda.cu            # Implementação CUDA (GPU)
├── run_cuda_experiments.ps1     # Script de compilação e execução
├── compare_cuda_results.py      # Validação de corretude
├── README.md                    # Este arquivo
└── dados.csv                    # Dados de teste (gerado)
    centroides_iniciais.csv      # Centróides iniciais (gerado)
```

## 🚀 Como Usar

### Pré-requisitos

```powershell
# Windows
# - GCC (MinGW)
# - CUDA Toolkit 11.0+ (inclui NVCC)
# - Python 3.x com NumPy

# Verificar CUDA
nvidia-smi

# Adicionar CUDA ao PATH (se necessário)
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
```

### 1. Execução Automática (Recomendado)

```powershell
cd Cuda
.\run_cuda_experiments.ps1
```

Este script:
- ✓ Gera/usa dados de teste
- ✓ Compila versão CPU (GCC)
- ✓ Compila versão CUDA (NVCC)
- ✓ Executa ambas as versões 3 vezes
- ✓ Calcula speedup
- ✓ Valida resultados

### 2. Compilação Manual

#### Versão Sequencial (CPU)
```bash
gcc -O3 -std=c99 kmeans_1d_seq.c -o kmeans_1d_seq.exe -lm
```

#### Versão CUDA (GPU)
```bash
# Detectar compute capability da GPU
nvidia-smi

# Compilar (exemplo para GeForce GTX 1660 Ti - sm_75)
nvcc -O3 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda.exe

# Outras opções de -arch:
# sm_50 = Maxwell (GTX 750, 960, 970, 980, etc)
# sm_60 = Pascal (GTX 1060, 1070, 1080, etc)
# sm_61 = Pascal (GTX Titan X, 1080 Ti, etc)
# sm_70 = Volta (Titan V, Tesla V100, etc)
# sm_75 = Turing (RTX 2060, 2070, 2080, GTX 1660, 1660 Ti, etc)
# sm_80 = Ampere (RTX 3060, 3070, 3080, 3090, etc)
# sm_86 = Ampere (RTX 3050, etc)
# sm_90 = Ada (RTX 4080, 4090, etc)
```

### 3. Executar Individualmente

#### Versão CPU
```bash
.\kmeans_1d_seq.exe dados.csv centroides_iniciais.csv 20
```

#### Versão GPU
```bash
.\kmeans_1d_cuda.exe dados.csv centroides_iniciais.csv 20
```

### 4. Validar Resultados

```bash
python compare_cuda_results.py
```

Verifica:
- Equivalência de atribuições
- Equivalência de centróides
- Equivalência de SSE

## 📖 Algoritmo Detalhado

### Assignment Step (GPU)

```cuda
__global__ void kernel_assignment(double *data, int N, double *centroids, int K,
                                   int *assignments, double *sse_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    // Cada thread processa um ponto
    double point = data[i];
    double min_dist = INFINITY;
    int best_cluster = 0;
    
    // Encontrar centróide mais próximo
    for (int k = 0; k < K; k++) {
        double diff = point - centroids[k];
        double dist = diff * diff;
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = k;
        }
    }
    
    assignments[i] = best_cluster;
    sse_array[i] = min_dist;  // Usado para redução de SSE
}
```

**Paralelização:**
- Grid: ⌈N / 256⌉ blocos de 256 threads
- Cada thread processa 1 ponto
- Complexidade: O(N × K)

### Update Step (GPU)

#### Passo 1: Acumular Somas (Paralelo)
```cuda
__global__ void kernel_update_partial(int *assignments, double *data, int N, int K,
                                       double *sum_global, int *count_global)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    int cluster = assignments[i];
    atomicAdd(&sum_global[cluster], data[i]);     // Operação atômica
    atomicAdd(&count_global[cluster], 1);
}
```

#### Passo 2: Calcular Novos Centróides (Paralelo)
```cuda
__global__ void kernel_update_centroids(double *centroids, double *sum_global,
                                         int *count_global, int K, double *data, int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    if (count_global[k] > 0) {
        centroids[k] = sum_global[k] / count_global[k];
    } else {
        centroids[k] = data[0];
    }
}
```

**Paralelização:**
- Kernel 1: ⌈N / 256⌉ blocos × 256 threads (acumular)
- Kernel 2: ⌈K / 256⌉ blocos × 256 threads (calcular)
- Usa operações atômicas para thread-safety

### Redução de SSE (GPU)

```cuda
__global__ void kernel_reduce_sse(double *sse_array, int N, double *sse_result)
{
    extern __shared__ double sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (idx < N) ? sse_array[idx] : 0.0;
    __syncthreads();
    
    // Redução em shared memory (tree reduction)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        sse_result[blockIdx.x] = sdata[0];
    }
}
```

## 📊 Esperado de Desempenho

### GPU: NVIDIA GeForce GTX 1660 Ti

| N | K | CPU | GPU | Speedup |
|---|---|-----|-----|---------|
| 10K | 20 | ~1ms | ~2ms | 0.5x |
| 100K | 20 | ~10ms | ~5ms | 2x |
| 1M | 20 | ~100ms | ~20ms | 5x |
| 5M | 20 | ~500ms | ~60ms | 8x |

**Observações:**
- Speedup é baixo para problemas pequenos (overhead CUDA domina)
- Speedup cresce com N (GPU explora paralelismo)
- Transferência PCI-E é sobrecarga importante

## 🔍 Validação de Corretude

### Atribuições
- Devem ser 100% idênticas (ou muito similares se pontos são equidistantes)
- Script verifica primeiras 10.000 atribuições

### Centróides
- Devem ser numericamente equivalentes (tolerância: 1e-5)
- Pode haver pequenas diferenças por ordem de operações em paralelo

### SSE (Sum of Squared Errors)
- Calculado a partir de atribuições + centróides
- Deve ter diferença relativa < 0.1%

## 📝 Arquivos de Saída

### CPU
- `assign_seq.csv`: Atribuições (N linhas, 1 inteiro por linha)
- `centroids_seq.csv`: Centróides finais (K linhas, 1 double por linha)

### GPU
- `assign_cuda.csv`: Atribuições (N linhas)
- `centroids_cuda.csv`: Centróides finais (K linhas)

## 🔧 Troubleshooting

### ERRO: "nvcc: command not found"
```powershell
# Adicionar CUDA ao PATH
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
```

### ERRO: "Device does not support this compute capability"
```bash
# Descobrir compute capability da GPU
nvidia-smi

# Usar o valor correto com -arch. Exemplo:
# Para GTX 1660 Ti (sm_75)
nvcc -O3 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda.exe
```

### GPU muito lenta (mais lenta que CPU)
- Normal para N < 100K
- Overhead CUDA domina para problemas pequenos
- Aumentar N para observar speedup

### Saída CUDA vazia/erros
```bash
# Verificar disponibilidade de GPU
nvidia-smi

# Testar com programa CUDA simples
cat > test_cuda.cu << 'EOF'
#include <stdio.h>
__global__ void kernel() { printf("GPU funciona!\n"); }
int main() { kernel<<<1,1>>>(); cudaDeviceSynchronize(); }
EOF
nvcc test_cuda.cu -o test_cuda
./test_cuda
```

## 📚 Referências

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Atomic Operations in CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)
- [Parallel Reduction](https://docs.nvidia.com/cuda/samples/1_Utilities/reduction/)

## 📋 Notas

- Versão CUDA usa `cudaEventElapsedTime()` para medição com precisão de GPU
- Versão CPU usa `get_time_ms()` que detecta Windows/Linux automaticamente
- Ambas garantem resultados determinísticos com mesma seed
- Transferência PCI-E (CPU ↔ GPU) é considerada no tempo total

## 👨‍💻 Autor

Implementação para disciplina de Programação Concorrente e Distribuída
