 /*
 * ============================================================
 * K-Means 1D - Gerador de Dados Unificado
 * ============================================================
 * 
 * Gera dados.csv e centroides_iniciais.csv
 * Versão standalone sem dependências externas
 * 
 * Uso: ./gen_data [N] [K] [seed]
 * Exemplo: ./gen_data 100000 20 42
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ===== GERADOR LINEAR CONGRUENCIAL ===== */
static unsigned long seed_state = 42;

double random_double() {
    seed_state = (seed_state * 1103515245 + 12345) & 0x7fffffff;
    return (double)seed_state / 0x7fffffff;
}

/* ===== BOX-MULLER PARA DISTRIBUIÇÃO NORMAL ===== */
double random_normal(double mean, double stddev) {
    double u1 = random_double();
    double u2 = random_double();
    
    // Evitar log(0)
    while (u1 < 1e-7) u1 = random_double();
    
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z0 * stddev;
}

/* ===== CRIAR DIRETÓRIO DE DADOS ===== */
void ensure_data_dir() {
    #ifdef _WIN32
        mkdir("../Data");
    #else
        mkdir("../Data", 0755);
    #endif
}

/* ===== FUNÇÃO PRINCIPAL ===== */
int main(int argc, char *argv[]) {
    if (argc != 6) {
        printf("Uso: %s <out_dados> <out_centroides> N K seed\n", argv[0]);
        printf("Ex: %s data/dados.csv data/centroides.csv 10000 4 42\n", argv[0]);
        return 1;
    }

    const char *out_dados = argv[1];
    const char *out_centroides = argv[2];
    int n_points = atoi(argv[3]);
    int n_clusters = atoi(argv[4]);
    unsigned long seed = atol(argv[5]);

    if (argc > 3) seed = atol(argv[3]);

    /* Validações */
    if (n_points <= 0 || n_clusters <= 0) {
        fprintf(stderr, "Erro: N e K devem ser positivos\n");
        return 1;
    }

    seed_state = seed;

    printf("=== Gerador de Dados K-Means 1D ===\n");
    printf("Número de pontos (N): %d\n", n_points);
    printf("Número de clusters (K): %d\n", n_clusters);
    printf("Seed: %lu\n\n", seed);

    /* Criar diretório se necessário */
    ensure_data_dir();

    /* Definir centros dos clusters */
    double *cluster_centers = (double *)malloc(n_clusters * sizeof(double));
    if (!cluster_centers) {
        fprintf(stderr, "Erro: Sem memória para centros\n");
        return 1;
    }

    for (int i = 0; i < n_clusters; i++) {
        cluster_centers[i] = (100.0 * i) / n_clusters;
    }

    /* Gerar dados */
    double *data = (double *)malloc(n_points * sizeof(double));
    if (!data) {
        fprintf(stderr, "Erro: Sem memória para dados\n");
        free(cluster_centers);
        return 1;
    }

    int points_per_cluster = n_points / n_clusters;
    int idx = 0;

    /* Gerar pontos por cluster com distribuição normal */
    for (int c = 0; c < n_clusters; c++) {
        for (int p = 0; p < points_per_cluster; p++) {
            data[idx++] = random_normal(cluster_centers[c], 5.0);
        }
    }

    /* Adicionar pontos restantes ao último cluster */
    while (idx < n_points) {
        data[idx++] = random_normal(cluster_centers[n_clusters - 1], 5.0);
    }

    /* Embaralhar dados (Fisher-Yates) */
    for (int i = n_points - 1; i > 0; i--) {
        int j = (int)(random_double() * (i + 1));
        double temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }

    /* Salvar dados em out_dados */
    FILE *fp = fopen(out_dados, "w");
    if (!fp) {
        fprintf(stderr, "Erro: Impossível criar %s\n", out_dados);
        free(data);
        free(cluster_centers);
        return 1;
    }

    for (int i = 0; i < n_points; i++) {
        fprintf(fp, "%.10f\n", data[i]);
    }
    fclose(fp);

    /* Centróides iniciais (perturbados dos centros verdadeiros) */
    seed_state = seed;  /* Reset seed para reprodutibilidade */
    
    for (int i = 0; i < n_clusters; i++) {
        cluster_centers[i] = cluster_centers[i] + random_normal(0.0, 2.0);
    }

    /* Salvar centróides em out_centroides */
    FILE *fp2 = fopen(out_centroides, "w");
    if (!fp2) {
        fprintf(stderr, "Erro: Impossível criar %s\n", out_centroides);
        free(data);
        free(cluster_centers);
        return 1;
    }

    for (int k = 0; k < n_clusters; k++) {
        fprintf(fp2, "%.10f\n", cluster_centers[k]);
    }
    fclose(fp2);

    /* ===== ESTATÍSTICAS ===== */
    double min_val = data[0];
    double max_val = data[0];
    double sum_val = 0.0;

    for (int i = 0; i < n_points; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum_val += data[i];
    }

    double mean = sum_val / n_points;
    double variance = 0.0;

    for (int i = 0; i < n_points; i++) {
        variance += (data[i] - mean) * (data[i] - mean);
    }

    double stddev = sqrt(variance / n_points);

    printf("Arquivos gerados em ../Data/:\n");
    printf("  ✓ dados.csv (%d pontos)\n", n_points);
    printf("  ✓ centroides_iniciais.csv (%d centróides)\n\n", n_clusters);

    printf("Estatísticas dos dados:\n");
    printf("  Mínimo:      %.2f\n", min_val);
    printf("  Máximo:      %.2f\n", max_val);
    printf("  Média:       %.2f\n", mean);
    printf("  Desvio pad.: %.2f\n\n", stddev);

    printf("Centróides iniciais:\n");
    for (int i = 0; i < n_clusters; i++) {
        printf("  Cluster %2d: %.2f\n", i, cluster_centers[i]);
    }

    /* Limpeza */
    free(data);
    free(cluster_centers);

    return 0;
}
