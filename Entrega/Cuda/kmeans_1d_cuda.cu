/*
 * ============================================================
 * K-Means 1D - Versão CUDA (GPU)
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#ifdef _WIN32
    #include <direct.h>
#endif

#define MAX_ITER 100
#define EPSILON 1e-6
#define MAX_K 256

/* Macro para checar erros CUDA */
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
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

/* ===== CRIAR DIRETÓRIO DE RESULTADOS ===== */
void ensure_results_dir() {
    #ifdef _WIN32
        _mkdir("results");
    #else
        mkdir("results", 0755);
    #endif
}

/* ===== MEMÓRIA CONSTANTE CUDA ===== */
__constant__ double constant_centroids[MAX_K];

/* ===== KERNELS CUDA ===== */
__global__ void kernel_assignment_optimized(double *data, int N, int K,
                                             int *assignments, double *sse_array) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double point = data[i];
    double min_dist = 1e308;
    int best_cluster = 0;

    /* Usar centróides da memória constante (rápido, em cache) */
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

__global__ void kernel_update_reduction(int *assignments, double *data, int N, int K,
                                        double *block_sums, int *block_counts) {
    extern __shared__ char shared_memory[];
    double *shared_sums = (double *)shared_memory;
    int *shared_counts = (int *)&shared_memory[K * sizeof(double)];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;

    /* Inicializar shared memory */
    if (tid < K) {
        shared_sums[tid] = 0.0;
        shared_counts[tid] = 0;
    }
    __syncthreads();

    /* Cada thread processa múltiplos pontos (stride) */
    for (int i = bid * block_size + tid; i < N; i += gridDim.x * block_size) {
        int cluster = assignments[i];
        atomicAdd(&shared_sums[cluster], data[i]);
        atomicAdd(&shared_counts[cluster], 1);
    }

    __syncthreads();

    /* Escrever resultados do bloco na memória global */
    if (tid < K) {
        atomicAdd(&block_sums[bid * K + tid], shared_sums[tid]);
        atomicAdd(&block_counts[bid * K + tid], shared_counts[tid]);
    }
}

/*
 * Kernel Update Centroids: reduz somas acumuladas por blocos
 */
__global__ void kernel_update_centroids_optimized(double *d_centroids,
                                                   double *block_sums, int *block_counts,
                                                   int K, int num_blocks, double *data, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    /* Reduzir somas e contagens de todos os blocos */
    double total_sum = 0.0;
    int total_count = 0;

    for (int b = 0; b < num_blocks; b++) {
        total_sum += block_sums[b * K + k];
        total_count += block_counts[b * K + k];
    }

    /* Calcular novo centróide */
    if (total_count > 0) {
        d_centroids[k] = total_sum / total_count;
    } else {
        d_centroids[k] = data[0];
    }
}

/* ===== FUNÇÕES UTILITÁRIAS ===== */

Dataset read_data(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
        exit(1);
    }

    int N = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) {
        N++;
    }

    rewind(fp);

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

double calculate_sse_host(double *sse_array, int N) {
    double sse = 0.0;
    for (int i = 0; i < N; i++) {
        sse += sse_array[i];
    }
    return sse;
}

/* ===== FUNÇÃO PRINCIPAL ===== */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s dados.csv centroides.csv K [max_iter] [eps] [assign_out] [centroids_out]\n", argv[0]);
        return 1;
    }

    const char *data_file = argv[1];
    const char *centroids_file = argv[2];
    int K = atoi(argv[3]);
    int max_iter = (argc >= 5) ? atoi(argv[4]) : 50;
    double eps = (argc >= 6) ? atof(argv[5]) : 1e-4;
    const char *assign_out = (argc >= 7) ? argv[6] : "assign.csv";
    const char *centroids_out = (argc >= 8) ? argv[7] : "centroids.csv";

    if (K <= 0 || max_iter <= 0 || eps <= 0.0) {
        fprintf(stderr, "Erro: K, maxiter e epsilon devem ser positivos\n");
        return 1;
    }

    if (K > MAX_K) {
        fprintf(stderr, "Erro: K (%d) excede máximo (%d)\n", K, MAX_K);
        return 1;
    }

    ensure_results_dir();

    printf("=== K-Means 1D - Versão CUDA (GPU) ===\n");
    printf("Parâmetros: K=%d, max_iter=%d, eps=%e\n\n", K, max_iter, eps);

    /* Verificar CUDA disponível */
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

    Dataset dataset = read_data(data_file);
    printf("Total de pontos (N): %d\n", dataset.N);

    KMeansModel model_host;
    model_host.K = K;
    model_host.centroids = read_centroids(centroids_file, K);
    model_host.assignments = (int *)malloc(dataset.N * sizeof(int));

    printf("K (clusters): %d\n", K);
    printf("Max iterações: %d\n", max_iter);
    printf("Epsilon: %e\n\n", eps);

    /* ===== ALOCAR MEMÓRIA GPU ===== */
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

    /* Copiar dados para GPU */
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

    /* ===== TESTE AUTOMÁTICO DE BLOCK SIZES ===== */
    printf("=== TESTE DE BLOCK SIZES ===\n");
    printf("Testando diferentes configurações de blocos...\n\n");

    int block_sizes[] = {32, 64, 128, 256, 512};
    int num_block_sizes = 5;
    int best_block_size = 256;
    float best_time = 1e20f;

    float block_size_times[5];

    for (int bs_idx = 0; bs_idx < num_block_sizes; bs_idx++) {
        int block_size = block_sizes[bs_idx];

        CUDA_CHECK(cudaMemcpy(d_centroids, model_host.centroids, K * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(constant_centroids, model_host.centroids,
                                      K * sizeof(double)));

        int grid_size = (dataset.N + block_size - 1) / block_size;

        cudaEvent_t test_start, test_stop;
        CUDA_CHECK(cudaEventCreate(&test_start));
        CUDA_CHECK(cudaEventCreate(&test_stop));
        CUDA_CHECK(cudaEventRecord(test_start));

        for (int iter = 0; iter < 5; iter++) {
            kernel_assignment_optimized<<<grid_size, block_size>>>(d_data, dataset.N, K,
                                                                     d_assignments, d_sse_array);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaEventRecord(test_stop));
        CUDA_CHECK(cudaEventSynchronize(test_stop));

        float test_time;
        CUDA_CHECK(cudaEventElapsedTime(&test_time, test_start, test_stop));
        test_time /= 5.0;

        block_size_times[bs_idx] = test_time;
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

    /* Salvar resultado dos testes */
    FILE *block_size_file = fopen("results/block_size_test.csv", "w");
    if (block_size_file) {
        fprintf(block_size_file, "block_size,time_ms\n");
        for (int bs_idx = 0; bs_idx < num_block_sizes; bs_idx++) {
            fprintf(block_size_file, "%d,%.6f\n", block_sizes[bs_idx], block_size_times[bs_idx]);
        }
        fclose(block_size_file);
    }

    /* ===== ALGORITMO K-MEANS ===== */
    int block_size = best_block_size;
    // If a block size was provided via command-line, use it and skip auto test
    int provided_block_size = 0;
    if (argc >= 9) {
        provided_block_size = atoi(argv[8]);
        if (provided_block_size > 0) {
            block_size = provided_block_size;
            printf("Using provided block size: %d\n", block_size);
        }
    }
    if (provided_block_size > 0) {
        // overwrite best_time variable for reporting consistency
        best_block_size = block_size;
        best_time = 0.0f;
    }
    int grid_size_N = (dataset.N + block_size - 1) / block_size;
    int num_blocks = grid_size_N;

    CUDA_CHECK(cudaMalloc((void**)&d_block_sums, num_blocks * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_block_counts, num_blocks * K * sizeof(int)));

    double *sse_array_host = (double *)malloc(dataset.N * sizeof(double));

    /* Zerar array SSE na GPU */
    CUDA_CHECK(cudaMemset(d_sse_array, 0, dataset.N * sizeof(double)));

    cudaEvent_t total_start, total_stop;
    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&total_stop));
    CUDA_CHECK(cudaEventRecord(total_start));

    double prev_sse = 1e308;
    int iter;

    printf("=== EXECUTANDO K-MEANS ===\n");
    printf("Block size: %d, Grid size: %d\n\n", block_size, grid_size_N);

    for (iter = 0; iter < max_iter; iter++) {
        /* ===== ASSIGNMENT STEP ===== */
        CUDA_CHECK(cudaMemcpyToSymbol(constant_centroids, model_host.centroids,
                                      K * sizeof(double)));

        kernel_assignment_optimized<<<grid_size_N, block_size>>>(d_data, dataset.N, K,
                                                                   d_assignments, d_sse_array);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(sse_array_host, d_sse_array, dataset.N * sizeof(double),
                              cudaMemcpyDeviceToHost));

        double sse = calculate_sse_host(sse_array_host, dataset.N);

        /* Verificar convergência */
        double sse_change = fabs(prev_sse - sse);
        double rel_change = sse_change / (prev_sse + 1e-10);

        printf("Iteração %d: SSE = %.10f (var_rel = %.10e)\n",
               iter + 1, sse, rel_change);

        if (rel_change < eps) {
            printf("Convergiu!\n");
            iter++;
            break;
        }

        prev_sse = sse;

        /* ===== UPDATE STEP ===== */
        CUDA_CHECK(cudaMemset(d_block_sums, 0, num_blocks * K * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_block_counts, 0, num_blocks * K * sizeof(int)));

        size_t shared_mem_size = K * (sizeof(double) + sizeof(int));

        kernel_update_reduction<<<num_blocks, block_size, shared_mem_size>>>(
            d_assignments, d_data, dataset.N, K, d_block_sums, d_block_counts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int grid_size_K = (K + block_size - 1) / block_size;

        kernel_update_centroids_optimized<<<grid_size_K, block_size>>>(
            d_centroids, d_block_sums, d_block_counts, K, num_blocks, d_data, dataset.N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(model_host.centroids, d_centroids, K * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(total_stop));
    CUDA_CHECK(cudaEventSynchronize(total_stop));

    float kernel_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, total_start, total_stop));

    /* ===== COPIAR ATRIBUIÇÕES FINAIS ===== */
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

    /* ===== ANÁLISE DE DESEMPENHO ===== */
    float total_time_ms = h2d_time_ms + kernel_time_ms + d2h_time_ms;
    double throughput = (dataset.N * iter) / (kernel_time_ms / 1000.0);

    printf("\n=== RESULTADOS ===\n");
    printf("Iterações executadas: %d\n", iter);
    printf("SSE final: %.10f\n", prev_sse);

    printf("\n--- Timing Detalhado ---\n");
    printf("Transfer H2D: %.3f ms\n", h2d_time_ms);
    printf("Kernels:     %.3f ms\n", kernel_time_ms);
    printf("Transfer D2H: %.3f ms\n", d2h_time_ms);
    printf("TOTAL:       %.3f ms\n", total_time_ms);

    printf("\n--- Métricas ---\n");
    printf("Throughput: %.2f M pontos/segundo\n", throughput / 1e6);
    printf("Tempo/iteração: %.3f ms\n", kernel_time_ms / iter);

    /* Salvar resultados */
    save_assignments(assign_out, model_host.assignments, dataset.N);
    save_centroids(centroids_out, model_host.centroids, K);
    printf("\nResultados salvos em results/\n");
    printf("\n=== SAÍDA PADRÃO ===\n");
    printf("N=%d K=%d max_iter=%d eps=%g block_size=%d\n", 
        dataset.N, K, max_iter, eps, best_block_size);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", 
        iter, prev_sse, total_time_ms);
    printf("H2D: %.2f ms | Kernel: %.2f ms | D2H: %.2f ms\n", 
        h2d_time_ms, kernel_time_ms, d2h_time_ms);

    /* ===== LIMPEZA ===== */
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_sse_array));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_counts));

    free(dataset.data);
    free(model_host.centroids);
    free(model_host.assignments);
    free(sse_array_host);

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));
    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(total_stop));

    return 0;
}
