# Prompt para Coding AI – Etapa 1 – OpenMP no Projeto K-Means 1D

Implemente uma versão paralelizada do algoritmo K-Means 1D (C99), utilizando OpenMP para CPU/memória compartilhada, com base neste contexto:

## Contexto do Projeto
Leia dois arquivos CSV (sem cabeçalho): `dados.csv` (N pontos, 1 por linha) e `centroides_iniciais.csv` (K valores iniciais).

Utilize uma estrutura sequencial padrão: a cada iteração:
1. **Assignment:** Para cada ponto, atribua ao centróide mais próximo (minimizando `(X[i]-C[c])²`), acumulando SSE (sum of squared errors).
2. **Update:** Para cada cluster, recalcule o centróide como a média dos seus pontos atribuídos (caso um cluster tenha zero pontos, copie `X[0]` para este centróide).
3. **Critério de Parada:** Pare quando a variação relativa do SSE for menor que `eps` ou após `max_iter` iterações.

## Paralelização com OpenMP

### O que paralelizar:
- **Laço de Assignment:** Paralelize o loop `for (i = 0; i < N; i++)` usando OpenMP.
- **Laço de Update:** Implemente duas estratégias:
  - **Opção A (Recomendada):** Use acumuladores por thread (`sum_thread[c]`, `cnt_thread[c]`) e reduza manualmente após a região paralela.
  - **Opção B (Comparação):** Use `#pragma omp critical` para atualizar somas e contagens (menos eficiente).

### Entradas/Saídas:
- **Entradas:** `dados.csv`, `centroides_iniciais.csv`
- **Saídas no Terminal:** Número de iterações, SSE final, tempo total (ms)
- **Saídas em Arquivo:** `assign.csv` (cluster de cada ponto), `centroids.csv` (centróides finais)

## Experimentos e Medições Obrigatórias

### Controle de Variáveis:
- **Mantenha CONSTANTES para todos os testes:**
  - Mesmos arquivos de entrada: `dados.csv` e `centroides_iniciais.csv`
  - Mesmos parâmetros: `max_iter`, `eps`, `K`, `N`
  - Mesmo ambiente de hardware e software
  - Mesma seed para geração de dados (quando aplicável)

- **Varie APENAS:**
  - Número de threads OpenMP: `T ∈ {1, 2, 4, 8, 16, ...}`

### Métricas:
- **Speedup:** `tempo_serial / tempo_OpenMP`
- **Escalonamento:** Teste diferentes números de threads
- **Otimização:** Varie o schedule (`static` vs `dynamic`) e `chunk_size` em OpenMP
- **Validação:** SSE não deve aumentar durante as iterações; resultados devem ser idênticos à versão serial

### Execução:
- Execute cada configuração múltiplas vezes (ex: 5 execuções)
- Use a média dos tempos para calcular o speedup
- Verifique a corretude comparando SSE finais e atribuições entre execuções

## Compilação
```bash
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm