/*
 * K-Means 1D - Versão Paralela com OpenMP
 * Implementação com paralelização dos passos de Assignment e Update
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define MAX_ITER 100
#define EPSILON 1e-6

typedef struct {
    double *data;
    int N;
} Dataset;

typedef struct {
    double *centroids;
    int *assignments;
    int K;
} KMeansModel;

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

// Assignment step: atribuir cada ponto ao centróide mais próximo (PARALELO OTIMIZADO)
double assignment_step_parallel(Dataset dataset, KMeansModel *model) {
    double sse = 0.0;
    const int N = dataset.N;
    const int K = model->K;
    double *data = dataset.data;
    double *centroids = model->centroids;
    int *assignments = model->assignments;
    
    // Usar chunk_size maior para reduzir overhead
    #pragma omp parallel for reduction(+:sse) schedule(static, 10000)
    for (int i = 0; i < N; i++) {
        double point = data[i];
        double min_dist = INFINITY;
        int best_cluster = 0;
        
        // Encontrar centróide mais próximo (loop interno otimizado)
        for (int k = 0; k < K; k++) {
            double diff = point - centroids[k];
            double dist = diff * diff;
            
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        
        assignments[i] = best_cluster;
        sse += min_dist;
    }
    
    return sse;
}

// Update step: recalcular centróides (PARALELO OTIMIZADO - Opção A: acumuladores por thread)
void update_step_parallel_optA(Dataset dataset, KMeansModel *model) {
    const int N = dataset.N;
    const int K = model->K;
    double *data = dataset.data;
    int *assignments = model->assignments;
    
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    
    // Alocar acumuladores por thread com padding para evitar false sharing
    const int PADDING = 8; // Cache line padding
    double *sum_thread = (double *)calloc(num_threads * K * PADDING, sizeof(double));
    int *count_thread = (int *)calloc(num_threads * K * PADDING, sizeof(int));
    
    // Acumular por thread com chunk size maior
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *my_sum = &sum_thread[tid * K * PADDING];
        int *my_count = &count_thread[tid * K * PADDING];
        
        #pragma omp for schedule(static, 10000) nowait
        for (int i = 0; i < N; i++) {
            int cluster = assignments[i];
            my_sum[cluster] += data[i];
            my_count[cluster]++;
        }
    }
    
    // Redução manual e cálculo dos novos centróides (paralelo se K for grande)
    #pragma omp parallel for if(K > 10) schedule(static)
    for (int k = 0; k < K; k++) {
        double total_sum = 0.0;
        int total_count = 0;
        
        for (int t = 0; t < num_threads; t++) {
            total_sum += sum_thread[t * K * PADDING + k];
            total_count += count_thread[t * K * PADDING + k];
        }
        
        if (total_count > 0) {
            model->centroids[k] = total_sum / total_count;
        } else {
            // Se cluster vazio, copiar primeiro ponto
            model->centroids[k] = data[0];
        }
    }
    
    free(sum_thread);
    free(count_thread);
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

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <dados.csv> <centroides_iniciais.csv> [K] [num_threads]\n", argv[0]);
        return 1;
    }

    const char *data_file = argv[1];
    const char *centroids_file = argv[2];
    int K = (argc >= 4) ? atoi(argv[3]) : 3;
    int num_threads = (argc >= 5) ? atoi(argv[4]) : omp_get_max_threads();

    // Configurar número de threads
    omp_set_num_threads(num_threads);

    printf("=== K-Means 1D - Versão Paralela (OpenMP) ===\n");
    printf("Threads: %d\n", num_threads);
    printf("Lendo dados...\n");

    // Ler dados
    Dataset dataset = read_data(data_file);
    printf("Total de pontos: %d\n", dataset.N);

    // Inicializar modelo
    KMeansModel model;
    model.K = K;
    model.centroids = read_centroids(centroids_file, K);
    model.assignments = (int *)malloc(dataset.N * sizeof(int));

    printf("K = %d clusters\n", K);
    printf("Max iterações: %d\n", MAX_ITER);
    printf("Epsilon: %e\n\n", EPSILON);

    // Medir tempo
    double start = omp_get_wtime();

    // Algoritmo K-Means
    double prev_sse = INFINITY;
    int iter;
    
    for (iter = 0; iter < MAX_ITER; iter++) {
        // Assignment (paralelo)
        double sse = assignment_step_parallel(dataset, &model);
        
        // Verificar convergência
        double sse_change = fabs(prev_sse - sse);
        double rel_change = sse_change / (prev_sse + 1e-10);
        
        printf("Iteração %d: SSE = %.10f (variação relativa = %.10e)\n", 
               iter + 1, sse, rel_change);
        
        if (rel_change < EPSILON) {
            printf("Convergiu!\n");
            iter++;
            break;
        }
        
        prev_sse = sse;
        
        // Update (paralelo - Opção A)
        update_step_parallel_optA(dataset, &model);
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;

    printf("\n=== Resultados ===\n");
    printf("Iterações executadas: %d\n", iter);
    printf("SSE final: %.10f\n", prev_sse);
    printf("Tempo total: %.3f ms\n", elapsed_ms);

    // Salvar resultados
    char assign_file[256], centroids_output[256];
    sprintf(assign_file, "assign_omp_%d.csv", num_threads);
    sprintf(centroids_output, "centroids_omp_%d.csv", num_threads);
    
    save_assignments(assign_file, model.assignments, dataset.N);
    save_centroids(centroids_output, model.centroids, K);
    printf("\nResultados salvos em: %s e %s\n", assign_file, centroids_output);

    // Limpar memória
    free(dataset.data);
    free(model.centroids);
    free(model.assignments);

    return 0;
}
