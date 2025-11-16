/*
 * K-Means 1D - Versão CUDA Otimizada (GPU)
 * Implementação paralela em GPU usando CUDA com múltiplas otimizações
 * - Memória constante para centróides
 * - Redução eficiente por blocos
 * - Parâmetros configuráveis
 * - Teste automático de block sizes
 * - Métricas de desempenho detalhadas
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_ITER 100
#define EPSILON 1e-6
#define MAX_K 256

// Macro para checar erros CUDA
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, error, cudaGetErrorName(error), cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

typedef struct {
    double *data;
    int N;
} Dataset;

typedef struct {
    double *centroids;
    int *assignments;
    int K;
} KMeansModel;

typedef struct {
    double h2d_time;
    double kernel_time;
    double d2h_time;
    double total_time;
    double throughput;
} PerformanceMetrics;

// ===================================
// MEMÓRIA CONSTANTE CUDA
// ===================================

__constant__ double constant_centroids[MAX_K];

// ===================================
// KERNELS CUDA OTIMIZADOS
// ===================================

/*
 * Kernel Assignment Otimizado: Usa memória constante para centróides
 * Cada thread processa um ponto
 */
__global__ void kernel_assignment_optimized(double *data, int N, int K,
                                             int *assignments, double *sse_array) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N) return;
    
    double point = data[i];
    double min_dist = 1e308;
    int best_cluster = 0;
    
    // Usar centróides da memória constante (rápido, em cache)
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

/*
 * Kernel Update Redução Otimizado: Calcula somas e contagens por bloco
 * Usa shared memory para redução eficiente
 * Reduz contenção de memória global vs atomicAdd
 */
__global__ void kernel_update_reduction(int *assignments, double *data, int N, int K,
                                         double *block_sums, int *block_counts) {
    extern __shared__ char shared_memory[];
    
    double *shared_sums = (double *)shared_memory;
    int *shared_counts = (int *)&shared_memory[K * sizeof(double)];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    
    // Inicializar shared memory
    if (tid < K) {
        shared_sums[tid] = 0.0;
        shared_counts[tid] = 0;
    }
    __syncthreads();
    
    // Cada thread processa múltiplos pontos (stride)
    for (int i = bid * block_size + tid; i < N; i += gridDim.x * block_size) {
        int cluster = assignments[i];
        
        // Usar operações atômicas apenas em shared memory (mais rápido)
        atomicAdd(&shared_sums[cluster], data[i]);
        atomicAdd(&shared_counts[cluster], 1);
    }
    __syncthreads();
    
    // Escrever resultados do bloco na memória global
    if (tid < K) {
        atomicAdd(&block_sums[bid * K + tid], shared_sums[tid]);
        atomicAdd(&block_counts[bid * K + tid], shared_counts[tid]);
    }
}

/*
 * Kernel para calcular novos centróides
 * Reduz as somas acumuladas por blocos
 */
__global__ void kernel_update_centroids_optimized(double *d_centroids, 
                                                   double *block_sums, int *block_counts,
                                                   int K, int num_blocks, double *data, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k >= K) return;
    
    // Reduzir somas e contagens de todos os blocos
    double total_sum = 0.0;
    int total_count = 0;
    
    for (int b = 0; b < num_blocks; b++) {
        total_sum += block_sums[b * K + k];
        total_count += block_counts[b * K + k];
    }
    
    // Calcular novo centróide
    if (total_count > 0) {
        d_centroids[k] = total_sum / total_count;
    } else {
        d_centroids[k] = data[0];
    }
}

// ===================================
// FUNÇÕES UTILITÁRIAS
// ===================================

// Função para ler dados do CSV
Dataset read_data(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
        exit(1);
    }

    // Contar número de linhas
    int N = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) {
        N++;
    }
    rewind(fp);

    // Alocar e ler dados
    double *data = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        if (fscanf(fp, "%lf", &data[i]) != 1) {
            fprintf(stderr, "Erro ao ler dados\n");
            exit(1);
        }
    }
    fclose(fp);

    Dataset dataset = {data, N};
    return dataset;
}

// Função para ler centróides iniciais
double* read_centroids(const char *filename, int K) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
        exit(1);
    }

    double *centroids = (double *)malloc(K * sizeof(double));
    for (int k = 0; k < K; k++) {
        if (fscanf(fp, "%lf", &centroids[k]) != 1) {
            fprintf(stderr, "Erro ao ler centróides\n");
            exit(1);
        }
    }
    fclose(fp);
    return centroids;
}

// Salvar atribuições
void save_assignments(const char *filename, int *assignments, int N) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Erro ao criar arquivo: %s\n", filename);
        return;
    }
    
    for (int i = 0; i < N; i++) {
        fprintf(fp, "%d\n", assignments[i]);
    }
    fclose(fp);
}

// Salvar centróides
void save_centroids(const char *filename, double *centroids, int K) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Erro ao criar arquivo: %s\n", filename);
        return;
    }
    
    for (int k = 0; k < K; k++) {
        fprintf(fp, "%.10f\n", centroids[k]);
    }
    fclose(fp);
}

// Calcular SSE no host (mais simples para 1D)
double calculate_sse_host(double *sse_array, int N) {
    double sse = 0.0;
    for (int i = 0; i < N; i++) {
        sse += sse_array[i];
    }
    return sse;
}

// ===================================
// FUNÇÃO PRINCIPAL
// ===================================

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <dados.csv> <centroides_iniciais.csv> [K] [max_iter] [eps]\n", argv[0]);
        printf("Exemplo: %s dados.csv centroides.csv 20 100 1e-6\n", argv[0]);
        return 1;
    }

    const char *data_file = argv[1];
    const char *centroids_file = argv[2];
    int K = (argc >= 4) ? atoi(argv[3]) : 3;
    int max_iter = (argc >= 5) ? atoi(argv[4]) : MAX_ITER;
    double eps = (argc >= 6) ? atof(argv[5]) : EPSILON;

    if (K > MAX_K) {
        fprintf(stderr, "Erro: K (%d) excede máximo (%d)\n", K, MAX_K);
        return 1;
    }

    printf("=== K-Means 1D - Versão CUDA Otimizada (GPU) ===\n");
    printf("Parâmetros: K=%d, max_iter=%d, eps=%e\n\n", K, max_iter, eps);

    // Verificar CUDA disponível
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Erro: Nenhuma GPU CUDA encontrada!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Capacidade de Compute: %d.%d\n", prop.major, prop.minor);
    printf("Max threads por bloco: %d\n\n", prop.maxThreadsPerBlock);

    printf("Lendo dados...\n");

    // Ler dados do host
    Dataset dataset = read_data(data_file);
    printf("Total de pontos (N): %d\n", dataset.N);

    // Inicializar modelo
    KMeansModel model_host;
    model_host.K = K;
    model_host.centroids = read_centroids(centroids_file, K);
    model_host.assignments = (int *)malloc(dataset.N * sizeof(int));

    printf("K (clusters): %d\n", K);
    printf("Max iterações: %d\n", max_iter);
    printf("Epsilon: %e\n\n", eps);

    // ===================================
    // ALOCAR MEMÓRIA GPU
    // ===================================

    double *d_data;
    int *d_assignments;
    double *d_sse_array;
    double *d_block_sums;
    int *d_block_counts;
    double *d_centroids;

    CUDA_CHECK(cudaMalloc((void**)&d_data, dataset.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_assignments, dataset.N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_sse_array, dataset.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, K * sizeof(double)));

    // Copiar dados para GPU
    cudaEvent_t h2d_start, h2d_stop;
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventRecord(h2d_start));

    CUDA_CHECK(cudaMemcpy(d_data, dataset.data, dataset.N * sizeof(double), 
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(h2d_stop));

    float h2d_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time_ms, h2d_start, h2d_stop));

    printf("Dados copiados para GPU em: %.3f ms\n\n", h2d_time_ms);

    // ===================================
    // TESTE AUTOMÁTICO DE BLOCK SIZES
    // ===================================

    printf("=== TESTE DE BLOCK SIZES ===\n");
    printf("Testando diferentes configurações de blocos...\n\n");

    int block_sizes[] = {32, 64, 128, 256, 512};
    int num_block_sizes = 5;
    int best_block_size = 256;
    float best_time = 1e20f;

    for (int bs_idx = 0; bs_idx < num_block_sizes; bs_idx++) {
        int block_size = block_sizes[bs_idx];
        
        // Teste rápido com 5 iterações
        CUDA_CHECK(cudaMemcpy(d_centroids, model_host.centroids, K * sizeof(double), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(constant_centroids, model_host.centroids, K * sizeof(double)));

        int grid_size = (dataset.N + block_size - 1) / block_size;

        cudaEvent_t test_start, test_stop;
        CUDA_CHECK(cudaEventCreate(&test_start));
        CUDA_CHECK(cudaEventCreate(&test_stop));
        CUDA_CHECK(cudaEventRecord(test_start));

        for (int iter = 0; iter < 5; iter++) {
            kernel_assignment_optimized<<<grid_size, block_size>>>(d_data, dataset.N, K,
                                                                   d_assignments, d_sse_array);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(test_stop));
        CUDA_CHECK(cudaEventSynchronize(test_stop));

        float test_time;
        CUDA_CHECK(cudaEventElapsedTime(&test_time, test_start, test_stop));
        test_time /= 5.0;  // Tempo médio

        printf("  Block size %3d: %.3f ms/iteração", block_size, test_time);
        
        if (test_time < best_time) {
            best_time = test_time;
            best_block_size = block_size;
            printf(" <- MELHOR\n");
        } else {
            printf("\n");
        }

        CUDA_CHECK(cudaEventDestroy(test_start));
        CUDA_CHECK(cudaEventDestroy(test_stop));
    }

    printf("\nMelhor block size: %d (%.3f ms)\n\n", best_block_size, best_time);

    // ===================================
    // ALGORITMO K-MEANS COM BLOCK SIZE OTIMIZADO
    // ===================================

    int block_size = best_block_size;
    int grid_size_N = (dataset.N + block_size - 1) / block_size;
    int num_blocks = grid_size_N;

    // Alocar memória para redução por bloco
    CUDA_CHECK(cudaMalloc((void**)&d_block_sums, num_blocks * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_block_counts, num_blocks * K * sizeof(int)));

    // Alocar array SSE no host
    double *sse_array_host = (double *)malloc(dataset.N * sizeof(double));

    cudaEvent_t total_start, total_stop;
    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&total_stop));
    CUDA_CHECK(cudaEventRecord(total_start));

    double prev_sse = 1e308;
    int iter;

    printf("=== EXECUTANDO K-MEANS ===\n");
    printf("Block size: %d, Grid size: %d\n\n", block_size, grid_size_N);

    for (iter = 0; iter < max_iter; iter++) {
        // ===== ASSIGNMENT STEP =====

        // Copiar centróides para memória constante
        CUDA_CHECK(cudaMemcpyToSymbol(constant_centroids, model_host.centroids, 
                                      K * sizeof(double)));

        // Executar kernel com memória constante
        kernel_assignment_optimized<<<grid_size_N, block_size>>>(d_data, dataset.N, K,
                                                                 d_assignments, d_sse_array);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copiar SSE para host e calcular
        CUDA_CHECK(cudaMemcpy(sse_array_host, d_sse_array, dataset.N * sizeof(double),
                              cudaMemcpyDeviceToHost));
        
        double sse = calculate_sse_host(sse_array_host, dataset.N);

        // Verificar convergência
        double sse_change = fabs(prev_sse - sse);
        double rel_change = sse_change / (prev_sse + 1e-10);

        printf("Iteração %d: SSE = %.10f (rel_change = %.10e)\n", 
               iter + 1, sse, rel_change);

        if (rel_change < eps) {
            printf("Convergiu!\n");
            iter++;
            break;
        }

        prev_sse = sse;

        // ===== UPDATE STEP =====

        // Zerar acumuladores
        CUDA_CHECK(cudaMemset(d_block_sums, 0, num_blocks * K * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_block_counts, 0, num_blocks * K * sizeof(int)));

        // Shared memory size: K doubles + K ints
        size_t shared_mem_size = K * (sizeof(double) + sizeof(int));

        // Kernel de redução por bloco
        kernel_update_reduction<<<num_blocks, block_size, shared_mem_size>>>(
            d_assignments, d_data, dataset.N, K, d_block_sums, d_block_counts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Kernel para calcular novos centróides
        int grid_size_K = (K + block_size - 1) / block_size;
        kernel_update_centroids_optimized<<<grid_size_K, block_size>>>(
            d_centroids, d_block_sums, d_block_counts, K, num_blocks, d_data, dataset.N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copiar centróides atualizados para host
        CUDA_CHECK(cudaMemcpy(model_host.centroids, d_centroids, K * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(total_stop));
    CUDA_CHECK(cudaEventSynchronize(total_stop));

    float kernel_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, total_start, total_stop));

    // ===== COPIAR ATRIBUIÇÕES FINAIS =====

    cudaEvent_t d2h_start, d2h_stop;
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));
    CUDA_CHECK(cudaEventRecord(d2h_start));

    CUDA_CHECK(cudaMemcpy(model_host.assignments, d_assignments, dataset.N * sizeof(int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));

    float d2h_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time_ms, d2h_start, d2h_stop));

    // ===================================
    // ANÁLISE DE DESEMPENHO
    // ===================================

    float total_time_ms = h2d_time_ms + kernel_time_ms + d2h_time_ms;
    double throughput = (dataset.N * iter) / (kernel_time_ms / 1000.0);

    printf("\n=== RESULTADOS ===\n");
    printf("Iterações executadas: %d\n", iter);
    printf("SSE final: %.10f\n", prev_sse);
    printf("\n--- Timing Detalhado ---\n");
    printf("Transfer H2D:     %.3f ms\n", h2d_time_ms);
    printf("Kernels:          %.3f ms\n", kernel_time_ms);
    printf("Transfer D2H:     %.3f ms\n", d2h_time_ms);
    printf("TOTAL:            %.3f ms\n", total_time_ms);
    printf("\n--- Métricas ---\n");
    printf("Throughput: %.2f pontos/segundo\n", throughput);
    printf("Throughput: %.2f M pontos/segundo\n", throughput / 1e6);
    printf("Tempo médio/iteração: %.3f ms\n", kernel_time_ms / iter);

    // Salvar resultados
    save_assignments("assign_cuda.csv", model_host.assignments, dataset.N);
    save_centroids("centroids_cuda.csv", model_host.centroids, K);
    printf("\nResultados salvos em: assign_cuda.csv e centroids_cuda.csv\n");

    // Salvar métricas em arquivo
    FILE *metrics_file = fopen("metrics_cuda.txt", "w");
    if (metrics_file) {
        fprintf(metrics_file, "=== MÉTRICAS DE DESEMPENHO CUDA ===\n");
        fprintf(metrics_file, "GPU: %s\n", prop.name);
        fprintf(metrics_file, "Compute Capability: %d.%d\n", prop.major, prop.minor);
        fprintf(metrics_file, "Block Size Otimizado: %d\n", block_size);
        fprintf(metrics_file, "Número de Pontos: %d\n", dataset.N);
        fprintf(metrics_file, "Número de Clusters: %d\n", K);
        fprintf(metrics_file, "Iterações: %d\n", iter);
        fprintf(metrics_file, "SSE Final: %.10f\n\n", prev_sse);
        fprintf(metrics_file, "Transfer H2D: %.3f ms\n", h2d_time_ms);
        fprintf(metrics_file, "Kernels: %.3f ms\n", kernel_time_ms);
        fprintf(metrics_file, "Transfer D2H: %.3f ms\n", d2h_time_ms);
        fprintf(metrics_file, "TOTAL: %.3f ms\n\n", total_time_ms);
        fprintf(metrics_file, "Throughput: %.2f M pontos/segundo\n", throughput / 1e6);
        fclose(metrics_file);
    }

    // Limpar memória GPU
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_sse_array));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_counts));

    // Limpar memória host
    free(dataset.data);
    free(model_host.centroids);
    free(model_host.assignments);
    free(sse_array_host);

    // Destruir eventos
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));
    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(total_stop));

    printf("\nMétricas salvas em: metrics_cuda.txt\n");

    return 0;
}
