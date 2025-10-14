#!/usr/bin/env python3
"""
Visualização dos resultados dos experimentos K-Means 1D
Gera gráficos de speedup, eficiência e tempo de execução
"""

import matplotlib.pyplot as plt
import numpy as np

# Dados dos experimentos
threads = np.array([1, 2, 4, 8, 16])
tempo_serial = 8916.4  # ms
tempos_paralelo = np.array([8762.8, 4592.4, 2912.8, 2114.4, 2100.8])  # ms

# Calcular métricas
speedup = tempo_serial / tempos_paralelo
eficiencia = (speedup / threads) * 100

# Criar figura com múltiplos subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Análise de Desempenho: K-Means 1D com OpenMP", fontsize=16, fontweight="bold"
)

# 1. Tempo de Execução
ax1 = axes[0, 0]
ax1.plot(threads, tempos_paralelo, "bo-", linewidth=2, markersize=8, label="Paralelo")
ax1.axhline(
    y=tempo_serial, color="r", linestyle="--", linewidth=2, label="Serial (baseline)"
)
ax1.set_xlabel("Número de Threads", fontsize=12)
ax1.set_ylabel("Tempo de Execução (ms)", fontsize=12)
ax1.set_title("Tempo de Execução vs Threads", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(threads)

# 2. Speedup
ax2 = axes[0, 1]
ax2.plot(threads, speedup, "go-", linewidth=2, markersize=8, label="Speedup Real")
ax2.plot(
    threads, threads / threads[0], "r--", linewidth=2, label="Speedup Ideal (Linear)"
)
ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
ax2.set_xlabel("Número de Threads", fontsize=12)
ax2.set_ylabel("Speedup", fontsize=12)
ax2.set_title("Speedup vs Threads", fontsize=13, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(threads)

# 3. Eficiência
ax3 = axes[1, 0]
ax3.plot(threads, eficiencia, "mo-", linewidth=2, markersize=8)
ax3.axhline(
    y=100, color="r", linestyle="--", linewidth=2, label="Eficiência Ideal (100%)"
)
ax3.set_xlabel("Número de Threads", fontsize=12)
ax3.set_ylabel("Eficiência (%)", fontsize=12)
ax3.set_title("Eficiência vs Threads", fontsize=13, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(threads)
ax3.set_ylim([0, 120])

# 4. Comparação de Tempos (Barras)
ax4 = axes[1, 1]
x_pos = np.arange(len(threads) + 1)
labels = ["Serial"] + [f"{t}T" for t in threads]
valores = np.concatenate(([tempo_serial], tempos_paralelo))
cores = ["red"] + ["blue"] * len(threads)

bars = ax4.bar(x_pos, valores, color=cores, alpha=0.7, edgecolor="black", linewidth=1.5)
ax4.set_xlabel("Configuração", fontsize=12)
ax4.set_ylabel("Tempo de Execução (ms)", fontsize=12)
ax4.set_title("Comparação de Tempos", fontsize=13, fontweight="bold")
ax4.set_xticks(x_pos)
ax4.set_xticklabels(labels)
ax4.grid(True, alpha=0.3, axis="y")

# Adicionar valores nas barras
for bar, valor in zip(bars, valores):
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{valor:.1f}ms",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig("resultados_kmeans.png", dpi=300, bbox_inches="tight")
print("Gráfico salvo: resultados_kmeans.png")

# Criar segundo gráfico: Tabela de Resultados
fig2, ax = plt.subplots(figsize=(10, 5))
ax.axis("tight")
ax.axis("off")

# Dados da tabela
table_data = [
    ["Configuração", "Tempo (ms)", "Speedup", "Eficiência (%)"],
    ["Serial", f"{tempo_serial:.0f}", "1.00", "100.0"],
]

for i, t in enumerate(threads):
    table_data.append(
        [
            f"{t} Threads",
            f"{tempos_paralelo[i]:.0f}",
            f"{speedup[i]:.3f}",
            f"{eficiencia[i]:.1f}",
        ]
    )

table = ax.table(
    cellText=table_data,
    cellLoc="center",
    loc="center",
    colWidths=[0.25, 0.25, 0.25, 0.25],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Estilizar cabeçalho
for i in range(4):
    table[(0, i)].set_facecolor("#4CAF50")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Estilizar linhas
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor("#f0f0f0")

plt.title(
    "Tabela de Resultados - K-Means 1D com OpenMP",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
plt.savefig("tabela_resultados.png", dpi=300, bbox_inches="tight")
print("Tabela salva: tabela_resultados.png")

# Criar gráfico de análise de convergência
fig3, ax = plt.subplots(figsize=(10, 6))

iterations = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
sse_values = np.array(
    [
        28527209,
        17754133,
        16504220,
        16076725,
        15850128,
        15410383,
        15039132,
        14902216,
        14791075,
        14720425,
        14676625,
        14651151,
        14635494,
        14625942,
        14620208,
    ]
)

ax.plot(iterations, sse_values, "bo-", linewidth=2, markersize=10, label="SSE")
ax.set_xlabel("Iteração", fontsize=12)
ax.set_ylabel("SSE (Sum of Squared Errors)", fontsize=12)
ax.set_title("Convergência do Algoritmo K-Means", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xticks(iterations)
ax.legend()

# Adicionar anotações
for i, (x, y) in enumerate(zip(iterations, sse_values)):
    ax.annotate(
        f"{y:.2e}", xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=9
    )

# Adicionar linha de convergência
ax.axhline(
    y=sse_values[-1],
    color="r",
    linestyle="--",
    linewidth=1,
    alpha=0.5,
    label="Valor de Convergência",
)

plt.tight_layout()
plt.savefig("convergencia.png", dpi=300, bbox_inches="tight")
print("Gráfico de convergência salvo: convergencia.png")

print("\n✅ Todos os gráficos foram gerados com sucesso!")
print("\nArquivos criados:")
print("  - resultados_kmeans.png")
print("  - tabela_resultados.png")
print("  - convergencia.png")
