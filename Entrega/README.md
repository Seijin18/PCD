PCD - K-means 1D (Entrega)
==========================

Resumo
-----
Este repositório implementa o algoritmo K-means 1D em C com várias variantes: serial, OpenMP, CUDA e MPI. Além das implementações, há utilitários para gerar datasets, validar resultados entre implementações e agregar resultados de experimentos.

Estrutura principal
-------------------
- `bin/` : executáveis gerados pelo `Makefile`.
- `Data/` : datasets (pequeno, médio, grande) usados nos testes.
- `results/` : logs e saídas geradas pelos executáveis e scripts de experimentos.
- `OpenMP/`, `Cuda/`, `MPI/`, `Tools/` : código-fonte das diferentes implementações e geradores.
- `experiments/` : scripts Python para validar, executar e agregar resultados (`validate.py`, `save_run_csv.py`, `plot_results.py`).

Requisitos
----------
- Windows:
	- Visual Studio (para compilar com MSVC) — se for usar `mpi-msvc`.
	- MS-MPI runtime + SDK (opcional, para `mpi-msvc`).
	- (Opcional) CUDA Toolkit (nvcc) para compilar a versão CUDA.
- Linux / WSL:
	- `gcc`, `make`.
	- `openmpi` / `mpicc` / `mpiexec` para executar e compilar MPI.
	- `nvcc` (CUDA Toolkit) se for usar a versão CUDA.
- Python 3.8+ para os scripts em `experiments/`.

Observação: se estiver no Windows e preferir um ambiente Unix-like, recomendo usar WSL2 + OpenMPI para facilitar compilação com `make`/`mpicc`/`nvcc`.

Build (usando o `Makefile`)
---------------------------
Todos os comandos abaixo devem ser executados no diretório `Entrega/` (onde está o `Makefile`) — por exemplo:

```
cd Entrega
make all
```

O alvo `all` compila as versões `serial`, `omp`, `cuda` e o gerador de dados (`gen_data`).

Principais alvos do `Makefile`
------------------------------
- `make all` : compila `serial`, `omp`, `cuda` e `gen_data`.
- `make serial` : compila apenas a versão serial (`bin/kmeans_serial`).
- `make omp` : compila apenas a versão OpenMP (`bin/kmeans_omp`).
- `make cuda` : compila apenas a versão CUDA (`bin/kmeans_cuda`) (requer `nvcc`).
- `make mpi` : compila a versão MPI usando a variável `MPICC` (default `mpicc`).
- `make mpi-msvc` (Windows): compila a versão MPI com MSVC e linka contra MS-MPI (requer Visual Studio Dev Tools e MS-MPI SDK). Este alvo chama `vcvarsall.bat` internamente.
- `make generate` / `generate-small|medium|large` : gera datasets em `Data/` (usar `gen_data`).
- `make test` / `test-serial|test-omp|test-cuda` : executa testes rápidos usando os datasets gerados.
- `make test-large` : executa uma bateria de testes para o dataset grande (gera logs em `results/`).
- `make benchmark-serial|benchmark-omp|benchmark-cuda|benchmark` : executa benchmarks conforme descrito no Makefile.
- `make clean` : remove diretório `bin/` e `results/`.
- `make clean-results` : remove apenas `results/`.
- `make distclean` : remove também os dados gerados em `Data/`.

Exemplos de execução
--------------------
Substitua os caminhos abaixo conforme o sistema (Windows usa `\\` em exemplos com `cmd`, mas em PowerShell prefira `./bin/...` ou `\\` conforme mostrado). Exemplos assumem que você está em `Entrega/`.

Serial:
```
./bin/kmeans_serial.exe Data/dados_small.csv Data/centroides_small.csv 4 50 1e-4 results/assign_serial.csv results/centroids_serial.csv
```

OpenMP (PowerShell):
```
$env:OMP_NUM_THREADS = 4
./bin/kmeans_omp.exe Data/dados_small.csv Data/centroides_small.csv 4 50 1e-4 results/assign_omp.csv results/centroids_omp.csv
```

OpenMP (cmd.exe):
```
set OMP_NUM_THREADS=4
bin\\kmeans_omp.exe Data\\dados_small.csv Data\\centroides_small.csv 4 50 1e-4 results\\assign_omp.csv results\\centroids_omp.csv
```

CUDA:
```
./bin/kmeans_cuda Data/dados_small.csv Data/centroides_small.csv 4 50 1e-4 results/assign_cuda.csv results/centroids_cuda.csv
```

MPI (com mpiexec):
```
mpiexec -n 4 ./bin/kmeans_mpi.exe Data/dados_large.csv Data/centroides_large.csv 16 50 1e-4 results/assign_mpi.csv results/centroids_mpi.csv
```

Se você compilou com `make mpi-msvc` no Windows, execute com:
```
mpiexec -n 4 bin\\kmeans_mpi.exe Data\\dados_large.csv Data\\centroides_large.csv 16 50 1e-4 results\\assign_mpi.csv results\\centroids_mpi.csv
```

Validação automática e scripts de experimentos
---------------------------------------------
O diretório `experiments/` contém scripts Python úteis:

- `validate.py` : valida correção entre implementações (compara SSE e assignments).
	Exemplo:
	```
	python experiments/validate.py --mpi-procs 4 --data Data/dados_large.csv --init Data/centroides_large.csv
	```

- `save_run_csv.py`, `plot_results.py` : ajudam a agregar logs de execução e gerar gráficos a partir de `results/run_summary.csv`.

Onde os arquivos de saída são salvos
----------------------------------
- Logs e saídas dos executáveis são colocados em `results/` (o `Makefile` cria e usa esse diretório).
- Executáveis ficam em `bin/`.

Dicas e troubleshooting
----------------------
- Se `make all` falhar no Windows por falta do `nvcc` (CUDA), rode `make serial omp` para compilar apenas as versões CPU.
- Se `make mpi` falhar no Windows por não existir `mpicc`, use `make mpi-msvc` a partir de um Developer Command Prompt do Visual Studio ou instale/usuário um wrapper `mpicc` em WSL.
- Para `mpi-msvc` é esperado que o MS-MPI SDK esteja em `C:\\\\Program Files (x86)\\\\Microsoft SDKs\\\\MPI\\\\` ou que você ajuste o `Makefile` com o caminho correto.
- No PowerShell, para definir variável OpenMP temporariamente para um comando, use:
	```powershell
	$env:OMP_NUM_THREADS = 4
	./bin/kmeans_omp.exe ...
	Remove-Item Env:\\\\OMP_NUM_THREADS
	```
- Em caso de problemas com paths longos no Windows, use a versão WSL para compilar e rodar os benchmarks (WSL tende a simplificar fluxos com `make`/`mpicc`/`nvcc`).