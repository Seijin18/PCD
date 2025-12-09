/*
 * ============================================================
 * K-Means 1D - Versão Serial (Baseline)
 * ============================================================
 * 
 * Implementação sequencial para comparação com paralelizadas
 * Otimizada com clock_gettime() para precisão de medição
 * 
 * Compilar: gcc -O2 -std=c99 -lm -o kmeans_serial kmeans_1d_serial.c
 * Usar: ./kmeans_serial ../data/dados.csv ../data/centroides_iniciais.csv K [maxiter] [eps]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

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

/* ===== CRIAR DIRETÓRIO DE RESULTADOS ===== */
void ensure_results_dir() {
    #ifdef _WIN32
        mkdir("results");
    #else
        mkdir("results", 0755);
    #endif
}

/* ===== MEDIÇÃO DE TEMPO DE ALTA PRECISÃO ===== */
double get_time_ms() {
    #ifdef _WIN32
        return ((double)clock() / CLOCKS_PER_SEC) * 1000.0;
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
    #endif
}

/* ===== FUNÇÕES UTILITÁRIAS DE I/O ===== */
Dataset read_data(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
        exit(1);
    }

    /* Contar número de linhas */
    int N = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) {
        N++;
    }

    rewind(fp);

    /* Alocar e ler dados */
    double *data = (double *)malloc(N * sizeof(double));
    if (!data) {
        fprintf(stderr, "Erro: Sem memória para dados\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        if (fscanf(fp, "%lf", &data[i]) != 1) {
            fprintf(stderr, "Erro ao ler dados na linha %d\n", i + 1);
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
    if (!centroids) {
        fprintf(stderr, "Erro: Sem memória para centróides\n");
        exit(1);
    }

    for (int k = 0; k < K; k++) {
        if (fscanf(fp, "%lf", &centroids[k]) != 1) {
            fprintf(stderr, "Erro ao ler centróide %d\n", k);
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

/* ===== ALGORITMO K-MEANS ===== */

/* Assignment step: atribuir cada ponto ao centróide mais próximo */
double assignment_step(Dataset dataset, KMeansModel *model) {
    double sse = 0.0;
    const int N = dataset.N;
    const int K = model->K;
    const double *data = dataset.data;
    const double *centroids = model->centroids;
    int *assignments = model->assignments;

    for (int i = 0; i < N; i++) {
        double point = data[i];
        double min_dist = INFINITY;
        int best_cluster = 0;

        /* Encontrar centróide mais próximo */
        for (int k = 0; k < K; k++) {
            double diff = point - centroids[k];
            double dist = diff * diff;  /* distância ao quadrado */

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

/* Update step: recalcular centróides */
void update_step(Dataset dataset, KMeansModel *model) {
    const int N = dataset.N;
    const int K = model->K;
    const double *data = dataset.data;
    const int *assignments = model->assignments;

    double *sum = (double *)calloc(K, sizeof(double));
    int *count = (int *)calloc(K, sizeof(int));

    if (!sum || !count) {
        fprintf(stderr, "Erro: Sem memória no update\n");
        exit(1);
    }

    /* Acumular soma e contagem por cluster */
    for (int i = 0; i < N; i++) {
        int cluster = assignments[i];
        sum[cluster] += data[i];
        count[cluster]++;
    }

    /* Calcular novos centróides */
    for (int k = 0; k < K; k++) {
        if (count[k] > 0) {
            model->centroids[k] = sum[k] / count[k];
        } else {
            /* Se cluster vazio, copiar primeiro ponto */
            model->centroids[k] = data[0];
        }
    }

    free(sum);
    free(count);
}

/* ===== FUNÇÃO PRINCIPAL ===== */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s dados.csv centroides.csv K [max_iter] [eps] [assign_out] [centroids_out]\n", argv[0]);
        printf("Ex: %s data/dados.csv data/centroides.csv 4 50 1e-4 assign.csv centroids.csv\n", argv[0]);
        return 1;
    }
    
    const char *data_file = argv[1];
    const char *centroids_file = argv[2];
    int K = atoi(argv[3]);
    int maxiter = (argc >= 5) ? atoi(argv[4]) : 50;
    double eps = (argc >= 6) ? atof(argv[5]) : 1e-4;
    const char *assign_out = (argc >= 7) ? argv[6] : "assign.csv";
    const char *centroids_out = (argc >= 8) ? argv[7] : "centroids.csv";

    /* Validações */
    if (K <= 0 || maxiter <= 0 || eps <= 0.0) {
        fprintf(stderr, "Erro: K, maxiter e epsilon devem ser positivos\n");
        return 1;
    }

    /* Criar diretório de resultados */
    ensure_results_dir();

    printf("=== K-Means 1D - Versão Serial (Baseline) ===\n");
    printf("Lendo dados...\n");

    /* Ler dados */
    Dataset dataset = read_data(data_file);
    printf("Total de pontos (N): %d\n", dataset.N);

    /* Inicializar modelo */
    KMeansModel model;
    model.K = K;
    model.centroids = read_centroids(centroids_file, K);
    model.assignments = (int *)malloc(dataset.N * sizeof(int));

    if (!model.assignments) {
        fprintf(stderr, "Erro: Sem memória para atribuições\n");
        return 1;
    }

    printf("Número de clusters (K): %d\n", K);
    printf("Max iterações: %d\n", maxiter);
    printf("Epsilon: %e\n\n", eps);

    /* ===== EXECUTAR K-MEANS ===== */
    double start = get_time_ms();

    double prev_sse = INFINITY;
    int iter;

    for (iter = 0; iter < maxiter; iter++) {
        /* Assignment */
        double sse = assignment_step(dataset, &model);

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

        /* Update */
        update_step(dataset, &model);
    }

    double end = get_time_ms();
    double elapsed_ms = end - start;

    /* ===== RESULTADOS ===== */
    printf("\n=== RESULTADOS ===\n");
    printf("Iterações executadas: %d\n", iter);
    printf("SSE final: %.10f\n", prev_sse);
    printf("Tempo total: %.3f ms\n", elapsed_ms);

    /* Salvar resultados em results/ */
    save_assignments(assign_out, model.assignments, dataset.N);
    save_centroids(centroids_out, model.centroids, K);
    printf("\n=== SAÍDA PADRÃO ===\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", dataset.N, K, maxiter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iter, prev_sse, elapsed_ms);

    /* Limpar memória */
    free(dataset.data);
    free(model.centroids);
    free(model.assignments);

    return 0;
}
