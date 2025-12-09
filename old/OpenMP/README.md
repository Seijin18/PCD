# K-Means 1D OpenMP - Entrega 1

Implementação paralela de K-Means 1D usando OpenMP com diferentes números de threads.

## 📋 Arquivos

### Código-Fonte
- `kmeans_1d_serial.c` - Versão sequencial (baseline)
- `kmeans_1d_omp.c` - Versão paralela com OpenMP

### Executáveis
- `kmeans_1d_serial.exe` - Versão sequencial compilada
- `kmeans_1d_omp.exe` - Versão paralela compilada

### Dados
- `dados.csv` - Dataset de entrada (100k pontos)
- `centroides_iniciais.csv` - Centróides iniciais (20 clusters)

### Scripts
- `generate_data.py` - Gerador de dados (standalone)
- `run_experiments.ps1` - Script para executar testes variando threads

## 🚀 Compilação

### Versão Sequencial
```bash
gcc -O3 -std=c99 -lm kmeans_1d_serial.c -o kmeans_1d_serial.exe
```

### Versão OpenMP (1, 2, 4, 8, 16 threads)
```bash
gcc -O3 -std=c99 -fopenmp -lm kmeans_1d_omp.c -o kmeans_1d_omp.exe
```

## ▶️ Execução

### Versão Sequencial
```powershell
.\kmeans_1d_serial.exe dados.csv centroides_iniciais.csv 20 100 1e-6
```

### Versão OpenMP
```powershell
$env:OMP_NUM_THREADS=4
.\kmeans_1d_omp.exe dados.csv centroides_iniciais.csv 20 100 1e-6
```

### Executar Todos os Testes
```powershell
.\run_experiments.ps1
```

## 📊 Parâmetros

| Parâmetro | Descrição |
|-----------|-----------|
| `dados.csv` | Arquivo de entrada com pontos |
| `centroides_iniciais.csv` | Centróides iniciais |
| `20` | Número de clusters (K) |
| `100` | Máximo de iterações |
| `1e-6` | Critério de convergência (epsilon) |

## 📈 Saídas

- `assign_omp_X.csv` - Atribuições finais (X = número de threads)
- `centroids_omp_X.csv` - Centróides finais (X = número de threads)

## ⚙️ Configuração OpenMP

Definir número de threads:
```powershell
$env:OMP_NUM_THREADS=4
```

## 📝 Formato de Dados

### dados.csv
```
ponto1
ponto2
...
ponto100000
```

### centroides_iniciais.csv
```
centroide1
centroide2
...
centroide20
```

## 🔄 Algoritmo K-Means

1. **Assignment**: Atribuir cada ponto ao centroide mais próximo
2. **Update**: Recalcular centróides como média dos pontos atribuídos
3. **Convergência**: Repetir até SSE variar menos que epsilon

## 🎯 Otimizações OpenMP

- Paralelização de loops no assignment step
- Redução paralela para acumulação de somas/contagens
- Critical sections para evitar race conditions
- Distribuição de carga balanceada entre threads

## 📊 Expectedado

- Speedup linear até ~8 threads (cores físicos)
- Saturação acima de 16 threads (hyperthreading)
- Overhead de sincronização em problemas pequenos

## 🔗 Referência

Entrega 1 - Comparativo: OpenMP vs CUDA
