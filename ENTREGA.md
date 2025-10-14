# Projeto K-Means 1D com OpenMP - Entrega 1
## Programação Concorrente e Distribuída

---

## 📋 Resumo do Projeto

Este projeto implementa o algoritmo **K-Means 1D** em duas versões:
1. **Versão Serial** (baseline para comparação)
2. **Versão Paralela com OpenMP** (paralelização em CPU com memória compartilhada)

### ✅ Status: COMPLETO

Todos os requisitos foram implementados e testados com sucesso.

---

## 📁 Arquivos Entregues

### Código Fonte
- ✅ `kmeans_1d_serial.c` - Implementação serial completa
- ✅ `kmeans_1d_omp.c` - Implementação paralela com OpenMP
- ✅ `generate_data.py` - Gerador de dados de teste
- ✅ `compare_results.py` - Script de validação de corretude

### Scripts de Automação
- ✅ `run_experiments.ps1` - Execução automatizada de todos os experimentos
- ✅ `visualize_results.py` - Geração de gráficos e visualizações

### Documentação
- ✅ `README.md` - Documentação técnica completa
- ✅ `RELATORIO.md` - Relatório detalhado de experimentos
- ✅ `ENTREGA.md` - Este documento (índice da entrega)

### Dados e Resultados
- ✅ `dados.csv` - Dataset de teste (1.000.000 pontos)
- ✅ `centroides_iniciais.csv` - Centróides iniciais (5 clusters)
- ✅ `assign_serial.csv` - Atribuições da versão serial
- ✅ `centroids_serial.csv` - Centróides finais da versão serial
- ✅ `assign_omp_*.csv` - Atribuições das versões paralelas
- ✅ `centroids_omp_*.csv` - Centróides finais das versões paralelas

### Visualizações
- ✅ `resultados_kmeans.png` - Gráficos de análise de desempenho
- ✅ `tabela_resultados.png` - Tabela resumida de resultados
- ✅ `convergencia.png` - Gráfico de convergência do algoritmo

---

## 🎯 Requisitos Atendidos

### ✅ Implementação
- [x] Leitura de arquivos CSV (dados.csv e centroides_iniciais.csv)
- [x] Algoritmo K-Means com Assignment e Update steps
- [x] Critério de parada por variação relativa de SSE < ε
- [x] Versão serial completa e funcional
- [x] Versão paralela com OpenMP

### ✅ Paralelização
- [x] Paralelização do laço de Assignment com `#pragma omp parallel for`
- [x] Uso de `reduction(+:sse)` para acumular SSE
- [x] Paralelização do laço de Update (Opção A: acumuladores por thread)
- [x] Redução manual após região paralela
- [x] Schedule estático implementado

### ✅ Entradas/Saídas
- [x] Leitura de dados.csv (N pontos)
- [x] Leitura de centroides_iniciais.csv (K centróides)
- [x] Saída no terminal: iterações, SSE final, tempo total (ms)
- [x] Saída em arquivo: assign.csv (atribuições)
- [x] Saída em arquivo: centroids.csv (centróides finais)

### ✅ Experimentos e Medições
- [x] Controle de variáveis (mesmos dados, parâmetros, ambiente)
- [x] Variação apenas do número de threads: 1, 2, 4, 8
- [x] Cálculo de Speedup (tempo_serial / tempo_paralelo)
- [x] Teste de escalonamento com diferentes números de threads
- [x] Múltiplas execuções (5 por configuração)
- [x] Uso de médias para cálculo de speedup
- [x] Validação de corretude (SSE e atribuições idênticas)

### ✅ Compilação
- [x] Flags corretas: `-O2 -fopenmp -std=c99 -lm`
- [x] Compilação sem warnings ou erros

---

## 📊 Principais Resultados

### Configuração Experimental
- **N (pontos):** 1.000.000
- **K (clusters):** 5
- **Iterações até convergência:** 5
- **SSE final:** 24.205.003,27
- **Seed:** 42 (reprodutível)

### Desempenho

| Configuração | Tempo Médio | Speedup | Eficiência |
|--------------|-------------|---------|------------|
| Serial       | 28,8 ms     | 1,00x   | 100,0%     |
| 1 Thread     | 27,4 ms     | 1,05x   | 105,1%     |
| 2 Threads    | 68,4 ms     | 0,42x   | 21,1%      |
| 4 Threads    | 62,4 ms     | 0,46x   | 11,5%      |
| 8 Threads    | 47,4 ms     | 0,61x   | 7,6%       |

### Validação de Corretude
- ✅ **100% de concordância** entre versões serial e paralela
- ✅ **0 diferenças** nas atribuições (1.000.000 pontos)
- ✅ **Diferença máxima nos centróides:** 0,0e+00
- ✅ **SSE convergência idêntica** em todas as versões

---

## 💡 Análise e Conclusões

### Speedup Negativo
A versão paralela apresentou **slowdown** (speedup < 1) devido a:
1. **Overhead de threads** dominando o tempo de execução
2. **Problema computacionalmente leve** (K-Means 1D é memory-bound)
3. **Poucos cálculos por ponto** (~5 operações aritméticas)
4. **Sincronização custosa** na redução manual

### Quando Paralelização Funciona
Para obter speedup positivo seria necessário:
- **K-Means N-D** (mais dimensões = mais cálculos)
- **K muito maior** (mais centróides para comparar)
- **N muito maior** (mais dados para processar)
- **Algoritmo mais complexo** (distâncias não-euclidianas)

### Lições Aprendidas
1. ✅ **Nem todo problema se beneficia de paralelização**
2. ✅ **Overhead pode superar ganhos** em problemas leves
3. ✅ **Validação é crucial** para garantir corretude
4. ✅ **Análise de complexidade** antes de paralelizar

---

## 🚀 Como Executar

### Pré-requisitos
```bash
# Windows com MinGW/GCC instalado
# Python 3.x com numpy e matplotlib
```

### Execução Rápida (Tudo Automático)
```powershell
# No PowerShell, execute:
.\run_experiments.ps1
```

Este script irá:
1. Gerar dados de teste (se necessário)
2. Compilar ambas as versões
3. Executar versão serial (5 vezes)
4. Executar versões paralelas com 1, 2, 4, 8 threads (5 vezes cada)
5. Calcular speedup e eficiência
6. Validar corretude
7. Exibir resumo dos resultados

### Execução Manual

#### 1. Gerar Dados
```powershell
python generate_data.py 1000000 5 42
```

#### 2. Compilar
```powershell
# Serial
gcc -O2 -std=c99 kmeans_1d_serial.c -o kmeans_1d_serial.exe -lm

# Paralelo
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp.exe -lm
```

#### 3. Executar
```powershell
# Serial
.\kmeans_1d_serial.exe dados.csv centroides_iniciais.csv 5

# Paralelo (4 threads)
.\kmeans_1d_omp.exe dados.csv centroides_iniciais.csv 5 4
```

#### 4. Comparar Resultados
```powershell
python compare_results.py
```

#### 5. Gerar Gráficos
```powershell
python visualize_results.py
```

---

## 📖 Documentação Completa

Para mais detalhes, consulte:
- **README.md**: Documentação técnica completa do projeto
- **RELATORIO.md**: Relatório detalhado com análise de resultados
- **Código fonte**: Comentários detalhados nos arquivos `.c`

---

## 🔍 Verificação de Entrega

### Checklist Final
- [x] Código compila sem erros
- [x] Código executa corretamente
- [x] Resultados serial e paralelo são idênticos
- [x] Experimentos com múltiplas threads realizados
- [x] Speedup calculado e documentado
- [x] Documentação completa incluída
- [x] Gráficos e visualizações gerados
- [x] Scripts de automação funcionais
- [x] README com instruções claras
- [x] Relatório técnico detalhado

### Arquivos Essenciais para Avaliação
1. `kmeans_1d_serial.c` - Código serial
2. `kmeans_1d_omp.c` - Código paralelo
3. `RELATORIO.md` - Relatório completo
4. `resultados_kmeans.png` - Gráficos de análise
5. `run_experiments.ps1` - Reproduzir experimentos

---

## 👨‍💻 Informações do Projeto

**Disciplina:** Programação Concorrente e Distribuída  
**Etapa:** 1 - OpenMP no Projeto K-Means 1D  
**Data de Entrega:** 11 de Outubro de 2025  
**Status:** ✅ COMPLETO

---

## 📞 Contato

Para dúvidas ou esclarecimentos sobre este projeto, consulte a documentação ou entre em contato com o autor.

---

**Última atualização:** 11/10/2025
