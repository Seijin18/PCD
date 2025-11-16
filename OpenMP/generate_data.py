#!/usr/bin/env python3
"""
Gerador de dados para o projeto K-Means 1D
Gera dados.csv e centroides_iniciais.csv
"""

import numpy as np
import sys


def generate_clustered_data(n_points=10000, n_clusters=3, seed=42):
    """
    Gera dados 1D com clusters bem definidos
    """
    np.random.seed(seed)

    # Definir centros dos clusters
    cluster_centers = np.linspace(0, 100, n_clusters)

    # Gerar pontos por cluster
    points_per_cluster = n_points // n_clusters
    data = []

    for center in cluster_centers:
        # Gerar pontos com distribuição normal ao redor do centro
        cluster_points = np.random.normal(center, 5.0, points_per_cluster)
        data.extend(cluster_points)

    # Adicionar pontos restantes ao último cluster se n_points não for divisível
    remaining = n_points - len(data)
    if remaining > 0:
        extra_points = np.random.normal(cluster_centers[-1], 5.0, remaining)
        data.extend(extra_points)

    # Embaralhar dados
    np.random.shuffle(data)

    return np.array(data), cluster_centers


def save_data(filename, data):
    """
    Salva dados em formato CSV (um valor por linha)
    """
    with open(filename, "w") as f:
        for value in data:
            f.write(f"{value:.10f}\n")


def save_centroids(filename, centroids):
    """
    Salva centróides iniciais em formato CSV
    """
    with open(filename, "w") as f:
        for centroid in centroids:
            f.write(f"{centroid:.10f}\n")


def main():
    # Parâmetros padrão
    n_points = 10000
    n_clusters = 3
    seed = 42

    # Permitir personalização via argumentos
    if len(sys.argv) > 1:
        n_points = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_clusters = int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])

    print(f"Gerando dados...")
    print(f"  Número de pontos: {n_points}")
    print(f"  Número de clusters: {n_clusters}")
    print(f"  Seed: {seed}")

    # Gerar dados
    data, true_centers = generate_clustered_data(n_points, n_clusters, seed)

    # Gerar centróides iniciais (perturbados dos centros verdadeiros)
    np.random.seed(seed)
    initial_centroids = true_centers + np.random.normal(0, 2.0, n_clusters)

    # Salvar arquivos
    save_data("dados.csv", data)
    save_centroids("centroides_iniciais.csv", initial_centroids)

    print(f"\nArquivos gerados:")
    print(f"  dados.csv ({len(data)} pontos)")
    print(f"  centroides_iniciais.csv ({len(initial_centroids)} centróides)")

    print(f"\nEstatísticas dos dados:")
    print(f"  Mínimo: {data.min():.2f}")
    print(f"  Máximo: {data.max():.2f}")
    print(f"  Média: {data.mean():.2f}")
    print(f"  Desvio padrão: {data.std():.2f}")

    print(f"\nCentróides iniciais:")
    for i, c in enumerate(initial_centroids):
        print(f"  Cluster {i}: {c:.2f}")


if __name__ == "__main__":
    main()
