/*
 * K-Means 1D - Versão Sequencial Otimizada (CPU)
 * Baseline para comparação com versão CUDA
 * Implementação eficiente usando alocação contígua e cache-friendly
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

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

    // Alocar e ler dados (alocação contígua para melhor cache)
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

// Assignment step: atribuir cada ponto ao centróide mais próximo
// Versão otimizada com cache de dados frequentemente acessados
double assignment_step(Dataset dataset, KMeansModel *model) {
    double sse = 0.0;
    const int N = dataset.N;
    const int K = model->K;
    double *data = dataset.data;
    double *centroids = model->centroids;
    int *assignments = model->assignments;
    
    // Loop otimizado para Assignment
    for (int i = 0; i < N; i++) {
        double point = data[i];
        double min_dist = INFINITY;
        int best_cluster = 0;
        
        // Encontrar centróide mais próximo
        for (int k = 0; k < K; k++) {
            double diff = point - centroids[k];
            double dist = diff * diff;  // distância ao quadrado
            
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

// Update step: recalcular centróides
// Versão otimizada com acumuladores contíguos
void update_step(Dataset dataset, KMeansModel *model) {
    const int N = dataset.N;
    const int K = model->K;
    
    double *sum = (double *)calloc(K, sizeof(double));
    int *count = (int *)calloc(K, sizeof(int));
    
    // Acumular soma e contagem por cluster (cache-friendly)
    for (int i = 0; i < N; i++) {
        int cluster = model->assignments[i];
        sum[cluster] += dataset.data[i];
        count[cluster]++;
    }
    
    // Calcular novos centróides
    for (int k = 0; k < K; k++) {
        if (count[k] > 0) {
            model->centroids[k] = sum[k] / count[k];
        } else {
            // Se cluster vazio, copiar primeiro ponto
            model->centroids[k] = dataset.data[0];
        }
    }
    
    free(sum);
    free(count);
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

// Medir tempo em ms (multiplataforma)
double get_time_ms() {
    #ifdef _WIN32
        // Windows: usar clock()
        return ((double)clock() / CLOCKS_PER_SEC) * 1000.0;
    #else
        // Linux/Mac: usar clock_gettime
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
    #endif
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <dados.csv> <centroides_iniciais.csv> [K]\n", argv[0]);
        return 1;
    }

    const char *data_file = argv[1];
    const char *centroids_file = argv[2];
    int K = (argc >= 4) ? atoi(argv[3]) : 3;

    printf("=== K-Means 1D - Versão Sequencial (CPU) ===\n");
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
    double start = get_time_ms();

    // Algoritmo K-Means
    double prev_sse = INFINITY;
    int iter;
    
    for (iter = 0; iter < MAX_ITER; iter++) {
        // Assignment
        double sse = assignment_step(dataset, &model);
        
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
        
        // Update
        update_step(dataset, &model);
    }

    double end = get_time_ms();
    double elapsed_ms = end - start;

    printf("\n=== Resultados ===\n");
    printf("Iterações executadas: %d\n", iter);
    printf("SSE final: %.10f\n", prev_sse);
    printf("Tempo total: %.3f ms\n", elapsed_ms);

    // Salvar resultados
    save_assignments("assign_seq.csv", model.assignments, dataset.N);
    save_centroids("centroids_seq.csv", model.centroids, K);
    printf("\nResultados salvos em: assign_seq.csv e centroids_seq.csv\n");

    // Limpar memória
    free(dataset.data);
    free(model.centroids);
    free(model.assignments);

    return 0;
}
