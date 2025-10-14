#!/usr/bin/env python3
"""
Script para comparar e validar resultados entre versões serial e paralela
"""

import sys
import numpy as np


def read_assignments(filename):
    """Lê arquivo de atribuições"""
    with open(filename, "r") as f:
        return [int(line.strip()) for line in f]


def read_centroids(filename):
    """Lê arquivo de centróides"""
    with open(filename, "r") as f:
        return [float(line.strip()) for line in f]


def compare_assignments(serial_file, parallel_file):
    """Compara atribuições entre serial e paralelo"""
    serial = read_assignments(serial_file)
    parallel = read_assignments(parallel_file)

    if len(serial) != len(parallel):
        print(
            f"ERRO: Tamanhos diferentes! Serial: {len(serial)}, Paralelo: {len(parallel)}"
        )
        return False

    differences = sum(1 for s, p in zip(serial, parallel) if s != p)
    total = len(serial)

    print(f"  Pontos totais: {total}")
    print(f"  Atribuições diferentes: {differences}")
    print(f"  Acurácia: {((total - differences) / total * 100):.2f}%")

    return differences == 0


def compare_centroids(serial_file, parallel_file, tolerance=1e-6):
    """Compara centróides entre serial e paralelo"""
    serial = np.array(read_centroids(serial_file))
    parallel = np.array(read_centroids(parallel_file))

    if len(serial) != len(parallel):
        print(
            f"ERRO: Número diferente de centróides! Serial: {len(serial)}, Paralelo: {len(parallel)}"
        )
        return False

    # Ordenar centróides para comparação (podem estar em ordem diferente)
    serial_sorted = np.sort(serial)
    parallel_sorted = np.sort(parallel)

    diff = np.abs(serial_sorted - parallel_sorted)
    max_diff = np.max(diff)

    print(f"  Número de centróides: {len(serial)}")
    print(f"  Diferença máxima: {max_diff:.10e}")
    print(f"  Tolerância: {tolerance:.10e}")

    print(f"\n  Centróides Serial (ordenados):")
    for i, c in enumerate(serial_sorted):
        print(f"    Cluster {i}: {c:.6f}")

    print(f"\n  Centróides Paralelo (ordenados):")
    for i, c in enumerate(parallel_sorted):
        print(f"    Cluster {i}: {c:.6f}")

    return max_diff < tolerance


def main():
    print("=" * 60)
    print("COMPARAÇÃO: Versão Serial vs Paralela")
    print("=" * 60)

    # Comparar atribuições
    print("\n1. Comparando ATRIBUIÇÕES...")
    print("-" * 60)

    try:
        threads_to_test = [1, 2, 4, 8]
        all_match = True

        for threads in threads_to_test:
            parallel_file = f"assign_omp_{threads}.csv"
            try:
                print(f"\nThreads = {threads}:")
                match = compare_assignments("assign_serial.csv", parallel_file)
                if match:
                    print("  ✓ MATCH: Atribuições idênticas!")
                else:
                    print("  ✗ DIFERENTE: Atribuições não coincidem!")
                    all_match = False
            except FileNotFoundError:
                print(f"  Arquivo não encontrado: {parallel_file}")
                continue

        if all_match:
            print("\n✓ Todas as atribuições coincidem!")
        else:
            print("\n✗ ATENÇÃO: Algumas atribuições diferem!")

    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado - {e}")
        return 1

    # Comparar centróides
    print("\n" + "=" * 60)
    print("2. Comparando CENTRÓIDES...")
    print("-" * 60)

    try:
        all_match = True

        for threads in threads_to_test:
            parallel_file = f"centroids_omp_{threads}.csv"
            try:
                print(f"\nThreads = {threads}:")
                match = compare_centroids("centroids_serial.csv", parallel_file)
                if match:
                    print("  ✓ MATCH: Centróides equivalentes (dentro da tolerância)!")
                else:
                    print("  ✗ DIFERENTE: Centróides não coincidem!")
                    all_match = False
            except FileNotFoundError:
                print(f"  Arquivo não encontrado: {parallel_file}")
                continue

        if all_match:
            print("\n✓ Todos os centróides coincidem!")
        else:
            print("\n✗ ATENÇÃO: Alguns centróides diferem!")

    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado - {e}")
        return 1

    print("\n" + "=" * 60)
    print("Comparação concluída!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
