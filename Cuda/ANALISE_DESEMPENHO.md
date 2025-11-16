# Análise de Desempenho - K-Means CUDA

## Sumário Executivo

**GPU:** NVIDIA GeForce GTX 1660 Ti

**Capacidade de Compute:** 7.5

**Throughput:** 126.58M pontos/segundo

**Tempo Total:** 79.404ms

## Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
| Block Size Otimizado | 128 threads |
| Número de Pontos | 100,000 |
| Número de Clusters | 20 |
| Iterações | 100 |
| Epsilon | 1e-6 |
| Max Iterações | 100 |

## Resultados de Performance

### Timing

- **Transfer H2D:** 0.171ms (0.2%)
- **Execução de Kernels:** 79.001ms (99.5%)
- **Transfer D2H:** 0.232ms (0.3%)
- **Tempo Total:** 79.404ms
- **Tempo Médio por Iteração:** 0.790ms

### Throughput

- **Throughput:** 126.58M pontos/segundo
- **Throughput (pontos/s):** 1.27e+08

### Convergência

- **SSE Final:** 1.9547123544e+06

## Análise de Block Sizes

| Block Size | Tempo (ms) |
|------------|------------|
| 32 | 0.126931 |
| 64 | 0.111155 |
| 128 | 0.111002 |
| 256 | 0.114637 |
| 512 | 0.125018 |

**Melhor Block Size:** 128 threads com 0.111002ms

**Melhoria vs Pior:** 12.5%

## Otimizações Implementadas

1. **Memória Constante para Centróides**
   - Centróides armazenados em memória constante (cache rápido)
   - Reduz latência de acesso à memória global

2. **Redução Otimizada por Blocos**
   - Uso de shared memory para redução eficiente
   - Minimiza contenção de atomicAdd em memória global

3. **Cálculo de SSE no Host**
   - SSE computado serialmente no host (adequado para 1D)
   - Elimina overhead de redução paralela complexa

4. **Teste Automático de Block Sizes**
   - Varredura de block sizes (32, 64, 128, 256, 512)
   - Seleção automática da configuração ótima

5. **Métricas de Desempenho Detalhadas**
   - Medição separada de transferências e execução
   - Cálculo de throughput em pontos/segundo

## Conclusões

- ✓ Implementação CUDA otimizada conforme especificação do projeto
- ✓ Teste automático de block sizes para melhor desempenho
- ✓ Métricas de desempenho detalhadas e exportadas
- ✓ Validação de corretude contra versão sequencial
- ✓ Gráficos gerados para análise visual

## Arquivos Gerados

- `metrics_cuda.csv` - Métricas detalhadas em formato CSV
- `metrics_cuda.txt` - Métricas em formato texto
- `block_size_test.csv` - Resultados de teste de block sizes
- `validation_cuda.txt` - Relatório de validação com versão sequencial
- `graphs/` - Diretório com todos os gráficos gerados
