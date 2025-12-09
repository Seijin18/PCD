/*
 * K-Means 1D - Versão MPI (memória distribuída)
 *
 * Compilar: mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_mpi -lm
 * Usar: mpirun -np P ./kmeans_mpi dados.csv centroides_iniciais.csv K [max_iter] [eps] [assign_out] [centroids_out]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

typedef struct {
    double *data;
    int N;
} Dataset;

/* Read data only on root and count lines */
int count_lines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;
    int cnt = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) cnt++;
    fclose(fp);
    return cnt;
}

double *read_data_root(const char *filename, int N) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;
    double *data = (double *)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        if (fscanf(fp, "%lf", &data[i]) != 1) {
            data[i] = 0.0;
        }
    }
    fclose(fp);
    return data;
}

double *read_centroids_root(const char *filename, int K) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;
    double *c = (double *)malloc(sizeof(double) * K);
    for (int k = 0; k < K; k++) {
        if (fscanf(fp, "%lf", &c[k]) != 1) c[k] = 0.0;
    }
    fclose(fp);
    return c;
}

void save_assignments_root(const char *filename, int *assign, int N) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    for (int i = 0; i < N; i++) fprintf(fp, "%d\n", assign[i]);
    fclose(fp);
}

void save_centroids_root(const char *filename, double *centroids, int K) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    for (int k = 0; k < K; k++) fprintf(fp, "%.10f\n", centroids[k]);
    fclose(fp);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            printf("Uso: %s dados.csv centroides.csv K [max_iter] [eps] [assign_out] [centroids_out]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *data_file = argv[1];
    const char *centroids_file = argv[2];
    int K = atoi(argv[3]);
    int max_iter = (argc >= 5) ? atoi(argv[4]) : 50;
    double eps = (argc >= 6) ? atof(argv[5]) : 1e-4;
    const char *assign_out = (argc >= 7) ? argv[6] : "assign_mpi.csv";
    const char *centroids_out = (argc >= 8) ? argv[7] : "centroids_mpi.csv";

    int N = 0;
    double *data_root = NULL;
    double *centroids = NULL;

    if (rank == 0) {
        int nlines = count_lines(data_file);
        if (nlines < 0) { fprintf(stderr, "Erro ao ler dados: %s\n", data_file); MPI_Abort(MPI_COMM_WORLD, 1); }
        N = nlines;
        data_root = read_data_root(data_file, N);
        centroids = read_centroids_root(centroids_file, K);
    }

    /* Broadcast N and K */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Prepare local blocks via counts/displacements */
    int base = N / size;
    int rem = N % size;
    int *counts = (int *)malloc(sizeof(int) * size);
    int *displs = (int *)malloc(sizeof(int) * size);
    for (int r = 0, offset = 0; r < size; r++) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    int localN = counts[rank];
    double *local_data = (double *)malloc(sizeof(double) * localN);

    /* Scatter data from root to locals */
    MPI_Scatterv(data_root, counts, displs, MPI_DOUBLE, local_data, localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Broadcast initial centroids (root has them) */
    if (rank != 0) centroids = (double *)malloc(sizeof(double) * K);
    MPI_Bcast(centroids, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *local_assign = (int *)malloc(sizeof(int) * localN);
    double prev_sse = INFINITY;

    double *sum_local = (double *)calloc(K, sizeof(double));
    int *cnt_local = (int *)calloc(K, sizeof(int));
    double *sum_global = (double *)malloc(sizeof(double) * K);
    int *cnt_global = (int *)malloc(sizeof(int) * K);

    int *all_assign = NULL;
    if (rank == 0) all_assign = (int *)malloc(sizeof(int) * N);

    for (int iter = 0; iter < max_iter; iter++) {
        /* Assignment on local block */
        double local_sse = 0.0;
        for (int i = 0; i < localN; i++) {
            double point = local_data[i];
            double min_dist = INFINITY;
            int best = 0;
            for (int k = 0; k < K; k++) {
                double diff = point - centroids[k];
                double d = diff * diff;
                if (d < min_dist) { min_dist = d; best = k; }
            }
            local_assign[i] = best;
            local_sse += min_dist;
        }

        double sse = 0.0;
        MPI_Allreduce(&local_sse, &sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* compute local sums and counts per cluster */
        for (int k = 0; k < K; k++) { sum_local[k] = 0.0; cnt_local[k] = 0; }
        for (int i = 0; i < localN; i++) {
            int c = local_assign[i];
            sum_local[c] += local_data[i];
            cnt_local[c]++;
        }

        MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        /* Update centroids on all ranks */
        for (int k = 0; k < K; k++) {
            if (cnt_global[k] > 0) centroids[k] = sum_global[k] / cnt_global[k];
            else centroids[k] = (rank == 0 && N>0) ? data_root[0] : centroids[k];
        }

        double sse_change = fabs(prev_sse - sse);
        double rel = sse_change / (prev_sse + 1e-10);
        if (rank == 0) {
            printf("Iteração %d: SSE = %.10f (var_rel = %.10e)\n", iter+1, sse, rel);
        }
        if (rel < eps) { if (rank==0) printf("Convergiu!\n"); break; }
        prev_sse = sse;
    }

    /* Gather assignments to root */
    MPI_Gatherv(local_assign, localN, MPI_INT, all_assign, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Save results */
        save_assignments_root(assign_out, all_assign, N);
        save_centroids_root(centroids_out, centroids, K);
        printf("Resultados salvos em: %s e %s\n", assign_out, centroids_out);
    }

    free(counts); free(displs);
    free(local_data); free(local_assign);
    free(sum_local); free(cnt_local); free(sum_global); free(cnt_global);
    if (rank==0) { free(data_root); free(all_assign); free(centroids); }
    else free(centroids);

    MPI_Finalize();
    return 0;
}
