/*
 * Gerador de dados para K-Means 1D
 * Gera dados.csv e centroides_iniciais.csv
 * Versão standalone sem dependências
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Gerador Linear Congruencial simples para números aleatórios
static unsigned long seed = 42;

double random_double() {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (double)seed / 0x7fffffff;
}

// Box-Muller para distribuição normal
double random_normal(double mean, double stddev) {
    double u1 = random_double();
    double u2 = random_double();
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z0 * stddev;
}

int main(int argc, char *argv[]) {
    int n_points = 100000;
    int n_clusters = 20;
    unsigned long random_seed = 42;

    // Parâmetros opcionais
    if (argc > 1) n_points = atoi(argv[1]);
    if (argc > 2) n_clusters = atoi(argv[2]);
    if (argc > 3) random_seed = atol(argv[3]);

    seed = random_seed;

    printf("Gerando dados...\n");
    printf("  Número de pontos: %d\n", n_points);
    printf("  Número de clusters: %d\n", n_clusters);
    printf("  Seed: %lu\n", random_seed);

    // Definir centros dos clusters
    double *cluster_centers = (double *)malloc(n_clusters * sizeof(double));
    for (int i = 0; i < n_clusters; i++) {
        cluster_centers[i] = (100.0 * i) / n_clusters;
    }

    // Gerar dados
    double *data = (double *)malloc(n_points * sizeof(double));
    int points_per_cluster = n_points / n_clusters;

    int idx = 0;
    for (int c = 0; c < n_clusters; c++) {
        for (int p = 0; p < points_per_cluster; p++) {
            data[idx++] = random_normal(cluster_centers[c], 5.0);
        }
    }

    // Adicionar pontos restantes
    while (idx < n_points) {
        data[idx++] = random_normal(cluster_centers[n_clusters - 1], 5.0);
    }

    // Embaralhar dados (Fisher-Yates)
    for (int i = n_points - 1; i > 0; i--) {
        int j = (int)(random_double() * (i + 1));
        double temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }

    // Salvar dados
    FILE *fp = fopen("dados.csv", "w");
    if (!fp) {
        fprintf(stderr, "Erro ao criar dados.csv\n");
        return 1;
    }
    for (int i = 0; i < n_points; i++) {
        fprintf(fp, "%.10f\n", data[i]);
    }
    fclose(fp);

    // Centróides iniciais (perturbados dos centros verdadeiros)
    seed = random_seed;  // Reset seed para reprodutibilidade
    for (int i = 0; i < n_clusters; i++) {
        cluster_centers[i] = cluster_centers[i] + random_normal(0.0, 2.0);
    }

    FILE *fp2 = fopen("centroides_iniciais.csv", "w");
    if (!fp2) {
        fprintf(stderr, "Erro ao criar centroides_iniciais.csv\n");
        return 1;
    }
    for (int k = 0; k < n_clusters; k++) {
        fprintf(fp2, "%.10f\n", cluster_centers[k]);
    }
    fclose(fp2);

    // Estatísticas
    double min_val = data[0], max_val = data[0], sum_val = 0.0;
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

    printf("\nArquivos gerados:\n");
    printf("  dados.csv (%d pontos)\n", n_points);
    printf("  centroides_iniciais.csv (%d centróides)\n", n_clusters);

    printf("\nEstatísticas dos dados:\n");
    printf("  Mínimo: %.2f\n", min_val);
    printf("  Máximo: %.2f\n", max_val);
    printf("  Média: %.2f\n", mean);
    printf("  Desvio padrão: %.2f\n", stddev);

    printf("\nCentróides iniciais:\n");
    for (int i = 0; i < n_clusters; i++) {
        printf("  Cluster %d: %.2f\n", i, cluster_centers[i]);
    }

    free(data);
    free(cluster_centers);

    return 0;
}
