OpenMP version — Compile and run
================================

Overview
--------
This folder contains the OpenMP parallel implementation of K-Means 1D (`kmeans_1d_omp.c`).

Compile
-------
From the project root (`Entrega`), the `Makefile` includes an `omp` target that compiles the OpenMP binary into `bin/`:

```powershell
Set-Location -Path 'D:\Projetinhos\Faculdade\PCD\Entrega 1\Entrega'
make omp
```

Or compile directly with GCC (Windows with MinGW/Cygwin or WSL):

```powershell
gcc -O3 -std=c99 -fopenmp OpenMP\kmeans_1d_omp.c -o bin\kmeans_omp.exe -lm
```

Run (PowerShell)
-----------------
Set the `OMP_NUM_THREADS` environment variable and run the executable. Example (PowerShell):

```powershell
$env:OMP_NUM_THREADS = 4
.\bin\kmeans_omp.exe Data\dados_small.csv Data\centroides_small.csv 4 50 1e-4 results\assign_omp_4_test.csv results\centroids_omp_4_test.csv
```

Measured quick tests (dataset: `Data/dados_small.csv`, K=4, max_iter=50/eps as above)
--------------------------------------------------------------------------
These are short runs executed locally to provide a baseline for speedup evaluation.

- Threads = 1  → Iterations: 5 | SSE final: 241957.5432408867 | Tempo: 38.0 ms
- Threads = 2  → Iterations: 5 | SSE final: 241957.5432408867 | Tempo: 31.0 ms
- Threads = 4  → Iterations: 5 | SSE final: 241957.5432408867 | Tempo: 23.0 ms
- Threads = 8  → Iterations: 5 | SSE final: 241957.5432408867 | Tempo: 25.0 ms
- Threads = 16 → Iterations: 5 | SSE final: 241957.5432408867 | Tempo: 26.0 ms

Notes
-----
- SSE and iteration counts match the serial implementation (within numerical tolerance) — good for correctness.
- Timing variability may occur due to OS scheduling, CPU frequency scaling and the small dataset size. For reproducible benchmarking, run multiple repetitions and use larger datasets (`generate-large`).
- The program may print the number of threads detected at runtime; verify `OMP_NUM_THREADS` or `omp_get_max_threads()` if results differ from expected.

Next steps
----------
- Automate repeated runs and collect CSV of timings for plotting (see `experiments/` in the todo list).
- Tune schedule (static/dynamic) and chunk size in the source to measure effects on performance for varied workloads.
