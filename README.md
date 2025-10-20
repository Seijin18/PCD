# Projeto K-Means 1D - OpenMP

Implementação **otimizada** do algoritmo K-Means 1D com paralelização OpenMP para a disciplina de Programação Concorrente e Distribuída.

## 🎯 Resultados Principais
- ✅ **Speedup:** 4.24x com 16 threads
- ✅ **Eficiência:** 97% com 2 threads, 76% com 4 threads
- ✅ **Dataset:** 5 milhões de pontos, 20 clusters
- ✅ **Corretude:** 100% validado

## Estrutura do Projeto

```
.
├── kmeans_1d_serial.c          # Versão serial (baseline)
├── kmeans_1d_omp.c              # Versão paralela com OpenMP
├── generate_data.py             # Gerador de dados de teste
├── run_experiments.ps1          # Script de execução dos experimentos
├── compare_results.py           # Validação de corretude
└── README.md                    # Este arquivo
```

## Compilação

### Versão Serial
```bash
gcc -O2 -std=c99 kmeans_1d_serial.c -o kmeans_1d_serial.exe -lm
```

### Versão Paralela (OpenMP)
```bash
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp.exe -lm
```

## Uso

### 1. Gerar Dados de Teste
```bash
python generate_data.py [num_pontos] [num_clusters] [seed]
```
Exemplo:
```bash
python generate_data.py 10000 3 42
```

Gera:
- `dados.csv`: Arquivo com N pontos (um por linha)
- `centroides_iniciais.csv`: Arquivo com K centróides iniciais

### 2. Executar Versão Serial
```bash
.\kmeans_1d_serial.exe dados.csv centroides_iniciais.csv 3
```

### 3. Executar Versão Paralela
```bash
.\kmeans_1d_omp.exe dados.csv centroides_iniciais.csv 3 [num_threads]
```
Exemplo com 4 threads:
```bash
.\kmeans_1d_omp.exe dados.csv centroides_iniciais.csv 3 4
```

### 4. Executar Todos os Experimentos
```powershell
.\run_experiments.ps1
```

Este script:
- Compila ambas as versões
- Gera dados de teste (se necessário)
- Executa versão serial 5 vezes
- Executa versão paralela com 1, 2, 4, 8 threads (5 vezes cada)
- Calcula speedup e eficiência
- Valida corretude comparando resultados

### 5. Comparar Resultados
```bash
python compare_results.py
```

## Arquivos de Saída

### Versão Serial
- `assign_serial.csv`: Atribuição de cluster para cada ponto
- `centroids_serial.csv`: Centróides finais

### Versão Paralela
- `assign_omp_[T].csv`: Atribuições (T = número de threads)
- `centroids_omp_[T].csv`: Centróides finais (T = número de threads)

## Algoritmo

### Assignment Step
Para cada ponto i:
1. Calcular distância ao quadrado para cada centróide k: `(X[i] - C[k])²`
2. Atribuir ao centróide mais próximo
3. Acumular SSE (Sum of Squared Errors)

**Paralelização:** Loop externo sobre pontos com `#pragma omp parallel for reduction(+:sse)`

### Update Step
Para cada cluster k:
1. Calcular soma e contagem dos pontos atribuídos
2. Novo centróide = soma / contagem
3. Se cluster vazio: copiar primeiro ponto

**Paralelização:** Acumuladores por thread (Opção A recomendada)
- Cada thread mantém somas/contagens locais
- Redução manual após região paralela

### Critério de Parada
Para quando:
- Variação relativa do SSE < ε (1e-6), ou
- Número máximo de iterações atingido (100)

## Métricas

### Speedup
```
S(T) = Tempo_Serial / Tempo_Paralelo(T)
```

### Eficiência
```
E(T) = S(T) / T × 100%
```

### Validação
- SSE final deve ser idêntico (ou muito próximo) entre serial e paralelo
- Atribuições devem ser idênticas
- SSE não deve aumentar durante iterações

## Experimentos Recomendados

1. **Escalabilidade Forte:** Fixar tamanho do problema, variar threads
   - N = 10,000 pontos, K = 3 clusters
   - T ∈ {1, 2, 4, 8, 16}

2. **Diferentes Schedules:** Testar `static` vs `dynamic`
   - Modificar `schedule(static)` para `schedule(dynamic, chunk_size)`

3. **Variação de Tamanho:** Diferentes N
   - N ∈ {1000, 10000, 100000}

## Requisitos

- GCC com suporte OpenMP
- Python 3.x (para geração de dados e comparação)
- NumPy (para scripts Python)

## Observações

- A versão paralela usa `omp_get_wtime()` para maior precisão temporal
- A versão serial usa `clock()` da stdlib
- Ambas as versões garantem resultados determinísticos com mesma seed
- SSE é calculado a cada iteração para monitoramento de convergência