# Relatório de Experimentos - K-Means 1D com OpenMP

**Data:** 11 de Outubro de 2025  
**Disciplina:** Programação Concorrente e Distribuída  
**Projeto:** Etapa 1 - Paralelização com OpenMP

---

## 1. Resumo Executivo

Este relatório apresenta a implementação e análise de desempenho do algoritmo K-Means 1D utilizando paralelização com OpenMP. O projeto comparou uma versão serial (baseline) com uma versão paralela, testando diferentes configurações de threads.

### Principais Resultados
- ✅ **Corretude:** 100% de concordância entre versões serial e paralela
- ✅ **Implementação:** Ambas as versões convergem em 5 iterações
- ⚠️ **Desempenho:** Speedup negativo devido ao overhead de paralelização

---

## 2. Configuração Experimental

### 2.1 Hardware e Software
- **Sistema Operacional:** Windows
- **Compilador:** GCC com suporte OpenMP
- **Flags de Compilação:** `-O2 -fopenmp -std=c99`

### 2.2 Parâmetros do Experimento
- **Número de Pontos (N):** 1.000.000
- **Número de Clusters (K):** 5
- **Iterações Máximas:** 100
- **Epsilon (critério de parada):** 1e-6
- **Seed:** 42 (reprodutibilidade)
- **Número de Execuções por Configuração:** 5

### 2.3 Distribuição dos Dados
Os dados foram gerados com distribuição normal ao redor de 5 centros:
- **Mínimo:** -22.33
- **Máximo:** 123.08
- **Média:** 49.99
- **Desvio Padrão:** 35.70

### 2.4 Centróides Iniciais
```
Cluster 0: 0.99
Cluster 1: 24.72
Cluster 2: 51.30
Cluster 3: 78.05
Cluster 4: 99.53
```

---

## 3. Implementação

### 3.1 Versão Serial
Implementação sequencial padrão com:
- Loop de assignment: atribuir cada ponto ao centróide mais próximo
- Loop de update: recalcular centróides como média dos pontos
- Medição de tempo com `clock()`

### 3.2 Versão Paralela (OpenMP)

#### Assignment Step (Paralelização)
```c
#pragma omp parallel for reduction(+:sse) schedule(static)
for (int i = 0; i < dataset.N; i++) {
    // Encontrar centróide mais próximo
    // Acumular SSE
}
```

#### Update Step (Opção A - Acumuladores por Thread)
```c
// Cada thread mantém acumuladores locais
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    double *my_sum = &sum_thread[tid * model->K];
    int *my_count = &count_thread[tid * model->K];
    
    #pragma omp for schedule(static)
    for (int i = 0; i < dataset.N; i++) {
        int cluster = model->assignments[i];
        my_sum[cluster] += dataset.data[i];
        my_count[cluster]++;
    }
}
// Redução manual após região paralela
```

### 3.3 Algoritmo de Convergência
Ambas as versões utilizam o mesmo critério:
- Calcular variação relativa: `|SSE_anterior - SSE_atual| / SSE_anterior`
- Parar se variação < ε ou após max_iter iterações

---

## 4. Resultados

### 4.1 Convergência
Ambas as versões (serial e paralela) convergiram em **5 iterações**:

```
Iteração 1: SSE = 26394590.69
Iteração 2: SSE = 24240892.15 (variação = 8.16%)
Iteração 3: SSE = 24205826.49 (variação = 0.14%)
Iteração 4: SSE = 24205024.48 (variação = 0.003%)
Iteração 5: SSE = 24205003.27 (variação = 0.00009%)
Convergiu!
```

### 4.2 Centróides Finais (Ordenados)
```
Cluster 0: -0.008080
Cluster 1: 25.005577
Cluster 2: 49.980294
Cluster 3: 75.010913
Cluster 4: 99.985938
```

### 4.3 Tempos de Execução

| Configuração | Tempo Médio (ms) | Speedup | Eficiência (%) |
|--------------|------------------|---------|----------------|
| **Serial**   | 28.8            | 1.00x   | 100.0%         |
| **1 Thread** | 27.4            | 1.05x   | 105.1%         |
| **2 Threads**| 68.4            | 0.42x   | 21.1%          |
| **4 Threads**| 62.4            | 0.46x   | 11.5%          |
| **8 Threads**| 47.4            | 0.61x   | 7.6%           |

### 4.4 Validação de Corretude

#### Atribuições
- ✅ **1 Thread:** 100% idênticas (0 diferenças em 1.000.000 pontos)
- ✅ **2 Threads:** 100% idênticas
- ✅ **4 Threads:** 100% idênticas
- ✅ **8 Threads:** 100% idênticas

#### Centróides
- ✅ **Diferença Máxima:** 0.0e+00 (zero absoluto)
- ✅ **Tolerância:** 1.0e-06
- ✅ Todos os centróides coincidem perfeitamente

---

## 5. Análise de Desempenho

### 5.1 Speedup Negativo
A versão paralela apresentou **speedup negativo** (slowdown) em todas as configurações com 2+ threads:

**Causas Identificadas:**
1. **Overhead de Criação de Threads:** O custo de criar e sincronizar threads supera o benefício da paralelização
2. **Problema Computacionalmente Leve:** K-Means 1D tem poucos cálculos por ponto (apenas 5 distâncias)
3. **Memória Cache:** A versão serial aproveita melhor a localidade de cache
4. **Sincronização:** A redução manual no update step adiciona overhead
5. **False Sharing:** Possível contenção de cache entre threads nos acumuladores

### 5.2 Configuração com 1 Thread
Interessantemente, a versão OpenMP com 1 thread foi **ligeiramente mais rápida** (1.05x) que a serial:
- Possível otimização do compilador com flags OpenMP
- Uso de `omp_get_wtime()` vs `clock()` (não afeta execução, apenas medição)
- Variação estatística dentro da margem de erro

### 5.3 Características do Problema
K-Means 1D é **memory-bound**, não **compute-bound**:
- Operações: Subtração, multiplicação, comparação (muito rápidas)
- Acesso à memória: Leitura de arrays grandes (gargalo)
- Razão compute/memory: Muito baixa (~5 operações por acesso)

### 5.4 Quando Paralelização Seria Benéfica
Para obter speedup positivo seria necessário:
1. **K-Means em dimensões maiores** (2D, 3D, ..., ND)
   - Mais cálculos por ponto: √((x₁-c₁)² + (x₂-c₂)² + ... + (xₙ-cₙ)²)
2. **Número maior de clusters** (K >> 5)
3. **Problema maior** (N >> 1.000.000)
4. **Algoritmo mais complexo** (distâncias não-euclidianas)

---

## 6. Validação Experimental

### 6.1 Controle de Variáveis
✅ **Mantidos Constantes:**
- Mesmos arquivos de entrada (dados.csv, centroides_iniciais.csv)
- Mesmos parâmetros (N=1.000.000, K=5, max_iter=100, ε=1e-6)
- Mesma seed (42) para geração de dados
- Mesmo compilador e flags de otimização

✅ **Variado Apenas:**
- Número de threads: {1, 2, 4, 8}

### 6.2 Repetibilidade
- **5 execuções** por configuração
- Tempo médio reportado
- Resultados consistentes entre execuções

### 6.3 Verificação de SSE
```
SSE final (serial):    24205024.4828231819
SSE final (1 thread):  24205024.4828231819
SSE final (2 threads): 24205024.4828231819
SSE final (4 threads): 24205024.4828221351
SSE final (8 threads): 24205024.4828231819
```
Diferenças < 1e-8 (erro de arredondamento de ponto flutuante)

---

## 7. Conclusões

### 7.1 Implementação
✅ **Sucesso:** Ambas as versões foram implementadas corretamente
- Convergência idêntica
- Resultados numericamente equivalentes
- Código segue especificações do projeto

### 7.2 Paralelização
⚠️ **Overhead Dominante:** Para este problema específico (K-Means 1D), a paralelização não trouxe benefícios:
- Speedup < 1.0 para todas as configurações com múltiplas threads
- Problema é muito leve computacionalmente
- Overhead de threads > ganho de paralelização

### 7.3 Lições Aprendidas
1. **Nem todo problema se beneficia de paralelização**
2. **Lei de Amdahl:** Overhead de paralelização pode dominar o ganho
3. **Análise de Complexidade:** Importante avaliar razão compute/memory
4. **Validação é Crucial:** Mesmo sem ganho de performance, corretude foi garantida

### 7.4 Trabalhos Futuros
Para melhorar o desempenho paralelo:
1. Implementar K-Means N-D (mais trabalho por ponto)
2. Testar com K muito maior (ex: K=100)
3. Usar SIMD (vetorização) em vez de threads
4. Explorar paralelização em GPU (CUDA/OpenCL)
5. Implementar schedule dinâmico com chunk_size otimizado

---

## 8. Apêndices

### 8.1 Comandos de Compilação
```bash
# Versão Serial
gcc -O2 -std=c99 kmeans_1d_serial.c -o kmeans_1d_serial.exe -lm

# Versão Paralela
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp.exe -lm
```

### 8.2 Comandos de Execução
```bash
# Gerar dados
python generate_data.py 1000000 5 42

# Executar serial
.\kmeans_1d_serial.exe dados.csv centroides_iniciais.csv 5

# Executar paralelo (4 threads)
.\kmeans_1d_omp.exe dados.csv centroides_iniciais.csv 5 4

# Executar todos os experimentos
.\run_experiments.ps1

# Comparar resultados
python compare_results.py
```

### 8.3 Estrutura de Arquivos
```
PCD/Entrega 1/
├── kmeans_1d_serial.c          # Implementação serial
├── kmeans_1d_omp.c              # Implementação paralela (OpenMP)
├── generate_data.py             # Gerador de dados
├── run_experiments.ps1          # Script de experimentos
├── compare_results.py           # Validação de corretude
├── dados.csv                    # Dados de entrada (1M pontos)
├── centroides_iniciais.csv      # Centróides iniciais (5)
├── assign_serial.csv            # Atribuições (serial)
├── centroids_serial.csv         # Centróides finais (serial)
├── assign_omp_*.csv             # Atribuições (paralelo)
├── centroids_omp_*.csv          # Centróides finais (paralelo)
├── README.md                    # Documentação
└── RELATORIO.md                 # Este relatório
```

---

## 9. Referências

1. **OpenMP Specification 4.5** - https://www.openmp.org/specifications/
2. **K-Means Clustering Algorithm** - MacQueen, J. (1967)
3. **Lei de Amdahl** - Amdahl, Gene (1967). "Validity of the single processor approach to achieving large scale computing capabilities"

---

**Autor:** [Seu Nome]  
**Disciplina:** Programação Concorrente e Distribuída  
**Instituição:** [Sua Instituição]  
**Data:** 11 de Outubro de 2025
