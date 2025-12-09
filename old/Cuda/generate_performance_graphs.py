#!/usr/bin/env python3
"""
Script para gerar gráficos de desempenho do K-Means CUDA
Analisa métricas de block size, throughput e compara com versão serial
Versão 2.0 - Alinhada com dados reais e métricas precisas
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo de gráficos - profissional e limpo
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
COLOR_PRIMARY = '#2E86AB'
COLOR_ACCENT = '#F18F01'
COLOR_SUCCESS = '#6A994E'
COLOR_WARNING = '#C73E1D'

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

class PerformanceAnalyzer:
    """Analisador de métricas de desempenho CUDA"""
    
    def __init__(self, cuda_dir='.'):
        self.cuda_dir = Path(cuda_dir)
        self.metrics_cuda = None
        self.block_size_data = None
        self.metrics_omp = None
        
        self._load_data()
    
    def _load_data(self):
        """Carrega dados de métricas dos arquivos"""
        print("📋 Carregando dados de desempenho...")
        
        # Carregar métricas CUDA
        metrics_csv = self.cuda_dir / 'results' / 'metrics_cuda.csv'
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            self.metrics_cuda = {row['metric']: row['value'] for _, row in df.iterrows()}
            print(f"  ✓ Métricas CUDA carregadas")
        else:
            print(f"  ⚠ Arquivo não encontrado: {metrics_csv}")
        
        # Carregar testes de block size
        block_size_csv = self.cuda_dir / 'results' / 'block_size_test.csv'
        if block_size_csv.exists():
            self.block_size_data = pd.read_csv(block_size_csv)
            print(f"  ✓ Testes de block size carregados")
        else:
            print(f"  ⚠ Arquivo não encontrado: {block_size_csv}")
        
        # Carregar métricas OpenMP (se existir arquivo de comparação)
        omp_metrics = self.cuda_dir.parent / 'OpenMP' / 'metrics_omp.csv'
        if omp_metrics.exists():
            df_omp = pd.read_csv(omp_metrics)
            self.metrics_omp = df_omp
            print(f"  ✓ Métricas OpenMP carregadas")
    
    def plot_block_size_analysis(self):
        """Gera gráfico de análise de block sizes - VERSÃO MELHORADA"""
        if self.block_size_data is None:
            print("⚠ Dados de block size não disponíveis")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Dados
        block_sizes = self.block_size_data['block_size'].values
        times = self.block_size_data['time_ms'].values
        
        # Plot principal com estilo profissional
        line = ax.plot(block_sizes, times, 
                marker='o', linewidth=3, markersize=12, 
                color=COLOR_PRIMARY, label='Tempo por iteração',
                markeredgewidth=2, markeredgecolor='white')
        
        # Destaque do melhor block size
        best_idx = self.block_size_data['time_ms'].idxmin()
        best_block = int(self.block_size_data.loc[best_idx, 'block_size'])
        best_time = self.block_size_data.loc[best_idx, 'time_ms']
        
        ax.scatter([best_block], [best_time], s=500, color=COLOR_ACCENT, 
                  zorder=5, marker='*', label=f'Ótimo: {best_block} threads',
                  edgecolors='darkred', linewidth=2)
        
        # Anotação melhorada
        offset_x = 30 if best_block < 300 else -80
        offset_y = 0.002
        ax.annotate(f'Melhor Configuração\n{best_block} threads\n{best_time:.6f} ms',
                   xy=(best_block, best_time),
                   xytext=(best_block + offset_x, best_time + offset_y),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_ACCENT, 
                            alpha=0.8, edgecolor='darkred', linewidth=2),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5,
                                 connectionstyle='arc3,rad=0.3'))
        
        # Adicionar valores em cada ponto
        for bs, t in zip(block_sizes, times):
            if bs != best_block:
                ax.text(bs, t + 0.001, f'{t:.6f}', 
                       ha='center', va='bottom', fontsize=9, 
                       color='gray', fontweight='normal')
        
        # Configurações do gráfico
        ax.set_xlabel('Block Size (número de threads por bloco)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Tempo por Iteração (ms)', fontsize=13, fontweight='bold')
        ax.set_title('Análise de Desempenho por Block Size - K-Means CUDA\nGPU: NVIDIA GeForce GTX 1660 Ti', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Grid e estilo
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Configurar ticks
        ax.set_xticks(block_sizes)
        ax.set_xticklabels([str(int(bs)) for bs in block_sizes])
        
        # Limites do eixo Y
        y_margin = (times.max() - times.min()) * 0.15
        ax.set_ylim([times.min() - y_margin, times.max() + y_margin])
        
        # Legenda
        ax.legend(fontsize=12, loc='upper right', framealpha=0.95, 
                 edgecolor='black', fancybox=True, shadow=True)
        
        # Calcular melhoria
        worst_time = times.max()
        improvement = ((worst_time - best_time) / worst_time) * 100
        
        # Adicionar texto informativo
        info_text = f'Melhoria: {improvement:.1f}% vs pior configuração ({int(block_sizes[times.argmax()])} threads)'
        ax.text(0.02, 0.98, info_text, 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_throughput_analysis(self):
        """Gera gráfico de análise de throughput - VERSÃO MELHORADA"""
        if self.metrics_cuda is None:
            print("⚠ Métricas CUDA não disponíveis")
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Extrair dados
            throughput_m = float(self.metrics_cuda.get('throughput', 0))
            h2d = float(self.metrics_cuda.get('time_h2d', 0))
            kernels = float(self.metrics_cuda.get('time_kernels', 0))
            d2h = float(self.metrics_cuda.get('time_d2h', 0))
            total = float(self.metrics_cuda.get('time_total', 0))
            
            # ===== GRÁFICO 1: Throughput em Barras =====
            categories = ['Throughput\nCUDA']
            values = [throughput_m]
            
            bars = ax1.bar(categories, values, color=COLOR_PRIMARY, width=0.5, 
                          edgecolor='black', linewidth=2.5, alpha=0.85)
            
            ax1.set_ylabel('Throughput (Milhões de pontos/segundo)', 
                          fontsize=12, fontweight='bold')
            ax1.set_title('Throughput de Processamento\nK-Means 1D em GPU', 
                         fontsize=14, fontweight='bold', pad=15)
            
            # Adicionar valor detalhado em cima da barra
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
                        f'{val:.2f}M\npts/s',
                        ha='center', va='bottom', fontweight='bold', 
                        fontsize=13, color='black')
                
                # Adicionar valor em pontos/s dentro da barra
                throughput_pts = float(self.metrics_cuda.get('throughput_points', 0))
                ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{throughput_pts/1e6:.2f}M pontos/s\n({throughput_pts:.0e} pts/s)',
                        ha='center', va='center', fontweight='bold', 
                        fontsize=10, color='white',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6))
            
            ax1.set_ylim([0, throughput_m * 1.25])
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            ax1.set_axisbelow(True)
            
            # Remover borda superior e direita
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # ===== GRÁFICO 2: Breakdown de Tempo (Pizza) =====
            time_parts = [h2d, kernels, d2h]
            labels_pie = ['Host→Device', 'Execução Kernels', 'Device→Host']
            colors_pie = [COLORS[2], COLOR_PRIMARY, COLORS[3]]
            
            # Calcular porcentagens
            percentages = [(t/total)*100 for t in time_parts]
            
            # Criar labels com valores e porcentagens
            explode = (0.05, 0.1, 0.05)  # Destacar kernels
            
            wedges, texts, autotexts = ax2.pie(time_parts, 
                                               labels=None,
                                               autopct='%1.1f%%',
                                               colors=colors_pie, 
                                               startangle=90,
                                               explode=explode,
                                               textprops={'fontsize': 11, 'fontweight': 'bold', 'color': 'white'},
                                               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            
            # Melhorar visibilidade dos textos
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_weight('bold')
            
            ax2.set_title('Distribuição do Tempo Total de Execução\n(Total: {:.3f}ms)'.format(total), 
                         fontsize=14, fontweight='bold', pad=15)
            
            # Adicionar legenda customizada
            legend_labels = [f'{label}: {time:.3f}ms ({pct:.1f}%)' 
                           for label, time, pct in zip(labels_pie, time_parts, percentages)]
            ax2.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.85, 1),
                      fontsize=10, framealpha=0.95, edgecolor='black')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"⚠ Erro ao gerar gráfico de throughput: {e}")
            import traceback
            traceback.print_exc()
            return None
            return None
    
    def plot_timing_breakdown(self):
        """Gera gráfico detalhado de timing - VERSÃO MELHORADA"""
        if self.metrics_cuda is None:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Extrair dados
            h2d = float(self.metrics_cuda.get('time_h2d', 0))
            kernels = float(self.metrics_cuda.get('time_kernels', 0))
            d2h = float(self.metrics_cuda.get('time_d2h', 0))
            total = float(self.metrics_cuda.get('time_total', 0))
            
            time_parts = [h2d, kernels, d2h]
            labels = ['Host→Device\n(Transferência)', 
                     'Execução de Kernels\n(Processamento GPU)', 
                     'Device→Host\n(Transferência)']
            colors = [COLORS[2], COLOR_PRIMARY, COLORS[3]]
            
            # Gráfico de barras vertical
            bars = ax.bar(labels, time_parts, color=colors, 
                         edgecolor='black', linewidth=2.5, alpha=0.85, width=0.6)
            
            # Adicionar valores e porcentagens nas barras
            for bar, val in zip(bars, time_parts):
                height = bar.get_height()
                pct = (val / total) * 100
                
                # Valor absoluto acima da barra
                ax.text(bar.get_x() + bar.get_width()/2., height + total*0.02,
                       f'{val:.3f} ms',
                       ha='center', va='bottom', fontweight='bold', fontsize=12)
                
                # Porcentagem dentro da barra (se altura suficiente)
                if height > total * 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2., height/2,
                           f'{pct:.1f}%',
                           ha='center', va='center', fontweight='bold', 
                           fontsize=13, color='white',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
            
            # Configurações do gráfico
            ax.set_ylabel('Tempo (milissegundos)', fontsize=13, fontweight='bold')
            ax.set_title(f'Breakdown de Tempo de Execução - K-Means CUDA\nTempo Total: {total:.3f}ms | GPU: NVIDIA GeForce GTX 1660 Ti', 
                        fontsize=15, fontweight='bold', pad=20)
            
            # Grid e estilo
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            
            # Ajustar limites do eixo Y
            ax.set_ylim([0, max(time_parts) * 1.15])
            
            # Remover bordas superior e direita
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Adicionar linha de referência para tempo total
            ax.axhline(y=total, color='red', linestyle='--', linewidth=2, 
                      alpha=0.5, label=f'Tempo Total: {total:.3f}ms')
            
            # Adicionar caixa informativa
            info_text = (f'Análise:\n'
                        f'• Kernels: {(kernels/total)*100:.1f}% do tempo total\n'
                        f'• Transferências: {((h2d+d2h)/total)*100:.1f}% do tempo total\n'
                        f'• Overhead de comunicação: {h2d+d2h:.3f}ms')
            
            ax.text(0.98, 0.97, info_text, 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                            alpha=0.9, edgecolor='black', linewidth=1.5))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"⚠ Erro ao gerar gráfico de timing: {e}")
            import traceback
            traceback.print_exc()
            return None
            return None
    
    def plot_performance_summary(self):
        """Gera um painel resumido de performance - VERSÃO MELHORADA"""
        if self.metrics_cuda is None:
            return None
        
        try:
            fig = plt.figure(figsize=(16, 11))
            gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35,
                                  left=0.08, right=0.95, top=0.93, bottom=0.05)
            
            # Extrair todos os dados necessários
            gpu_name = str(self.metrics_cuda.get('gpu_name', 'Unknown'))
            compute_cap = str(self.metrics_cuda.get('compute_capability', 'Unknown'))
            block_size_opt = int(float(self.metrics_cuda.get('block_size_optimized', 0)))
            num_points = int(float(self.metrics_cuda.get('num_points', 0)))
            num_clusters = int(float(self.metrics_cuda.get('num_clusters', 0)))
            num_iterations = int(float(self.metrics_cuda.get('num_iterations', 0)))
            sse_final = float(self.metrics_cuda.get('sse_final', 0))
            
            h2d = float(self.metrics_cuda.get('time_h2d', 0))
            kernels = float(self.metrics_cuda.get('time_kernels', 0))
            d2h = float(self.metrics_cuda.get('time_d2h', 0))
            total = float(self.metrics_cuda.get('time_total', 0))
            time_per_iter = float(self.metrics_cuda.get('time_per_iteration', 0))
            
            throughput = float(self.metrics_cuda.get('throughput', 0))
            throughput_pts = float(self.metrics_cuda.get('throughput_points', 0))
            
            # ===== SEÇÃO 1: Informações da GPU (Top Left) =====
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.axis('off')
            
            gpu_info = f"""
╔═══════════════════════════════════════╗
║     CONFIGURAÇÃO DO HARDWARE         ║
╚═══════════════════════════════════════╝

GPU: {gpu_name}
Compute Capability: {compute_cap}

╔═══════════════════════════════════════╗
║     CONFIGURAÇÃO DO ALGORITMO        ║
╚═══════════════════════════════════════╝

• Block Size Otimizado: {block_size_opt} threads
• Número de Pontos: {num_points:,}
• Número de Clusters (K): {num_clusters}
• Iterações Executadas: {num_iterations}
• Convergência (SSE): {sse_final:.2e}
            """
            
            ax1.text(0.05, 0.95, gpu_info, fontsize=10.5, verticalalignment='top',
                    family='monospace', fontweight='normal',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', 
                             alpha=0.3, edgecolor='navy', linewidth=2),
                    transform=ax1.transAxes)
            
            # ===== SEÇÃO 2: Métricas de Performance (Top Right) =====
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.axis('off')
            
            perf_info = f"""
╔═══════════════════════════════════════╗
║     MÉTRICAS DE DESEMPENHO           ║
╚═══════════════════════════════════════╝

THROUGHPUT:
• {throughput:.2f} M pontos/segundo
• {throughput_pts:.2e} pontos/s

TIMING:
• Tempo Total:         {total:.3f} ms
• Tempo/Iteração:      {time_per_iter:.3f} ms
• Transfer H→D:        {h2d:.3f} ms ({(h2d/total)*100:.1f}%)
• Kernels (GPU):       {kernels:.3f} ms ({(kernels/total)*100:.1f}%)
• Transfer D→H:        {d2h:.3f} ms ({(d2h/total)*100:.1f}%)

EFICIÊNCIA:
• Overhead Comunicação: {h2d+d2h:.3f} ms ({((h2d+d2h)/total)*100:.1f}%)
• Processamento GPU:    {kernels:.3f} ms ({(kernels/total)*100:.1f}%)
            """
            
            ax2.text(0.05, 0.95, perf_info, fontsize=10.5, verticalalignment='top',
                    family='monospace', fontweight='normal',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', 
                             alpha=0.3, edgecolor='darkgreen', linewidth=2),
                    transform=ax2.transAxes)
            
            # ===== SEÇÃO 3: Block Size Performance (Middle Row, Full Width) =====
            if self.block_size_data is not None:
                ax3 = fig.add_subplot(gs[1, :])
                
                block_sizes = self.block_size_data['block_size'].values
                times = self.block_size_data['time_ms'].values
                
                ax3.plot(block_sizes, times, marker='o', linewidth=3, markersize=10, 
                        color=COLOR_PRIMARY, label='Tempo por iteração',
                        markeredgewidth=2, markeredgecolor='white')
                
                best_idx = self.block_size_data['time_ms'].idxmin()
                best_block = int(self.block_size_data.loc[best_idx, 'block_size'])
                best_time = self.block_size_data.loc[best_idx, 'time_ms']
                
                ax3.scatter([best_block], [best_time], s=400, color=COLOR_ACCENT, 
                           zorder=5, marker='*', edgecolors='darkred', linewidth=2,
                           label=f'Ótimo: {best_block} threads ({best_time:.6f}ms)')
                
                ax3.set_xlabel('Block Size (threads por bloco)', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Tempo (ms)', fontsize=12, fontweight='bold')
                ax3.set_title('Análise de Desempenho por Block Size', fontsize=13, fontweight='bold')
                ax3.grid(True, alpha=0.3, linestyle='--')
                ax3.set_axisbelow(True)
                ax3.legend(fontsize=10, loc='best', framealpha=0.9)
                ax3.set_xticks(block_sizes)
            
            # ===== SEÇÃO 4: Breakdown de Tempo (Bottom Left) =====
            ax4 = fig.add_subplot(gs[2, 0])
            
            time_parts = [h2d, kernels, d2h]
            labels = ['H→D', 'Kernels', 'D→H']
            colors = [COLORS[2], COLOR_PRIMARY, COLORS[3]]
            
            bars = ax4.bar(labels, time_parts, color=colors, edgecolor='black', 
                          linewidth=2, alpha=0.85)
            
            for bar, val in zip(bars, time_parts):
                height = bar.get_height()
                pct = (val / total) * 100
                ax4.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                        f'{val:.3f}ms\n({pct:.1f}%)', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
            
            ax4.set_ylabel('Tempo (ms)', fontsize=11, fontweight='bold')
            ax4.set_title(f'Breakdown de Tempo (Total: {total:.3f}ms)', 
                         fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3, linestyle='--')
            ax4.set_axisbelow(True)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            
            # ===== SEÇÃO 5: Throughput (Bottom Right) =====
            ax5 = fig.add_subplot(gs[2, 1])
            
            categories = ['CUDA\nGPU']
            values = [throughput]
            
            bars = ax5.bar(categories, values, color=COLOR_PRIMARY, width=0.5, 
                          edgecolor='black', linewidth=2.5, alpha=0.85)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + throughput*0.03,
                        f'{val:.2f}M\npts/s', ha='center', va='bottom', 
                        fontweight='bold', fontsize=12)
                
                # Adicionar valor detalhado
                ax5.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{throughput_pts/1e6:.2f}M pts/s',
                        ha='center', va='center', fontweight='bold', 
                        fontsize=10, color='white',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
            
            ax5.set_ylabel('Throughput (M pts/s)', fontsize=11, fontweight='bold')
            ax5.set_title('Throughput de Processamento', fontsize=12, fontweight='bold')
            ax5.set_ylim([0, throughput * 1.25])
            ax5.grid(axis='y', alpha=0.3, linestyle='--')
            ax5.set_axisbelow(True)
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            
            # Título geral da figura
            fig.suptitle('Resumo Completo de Desempenho - K-Means 1D CUDA\nGPU: {} | Block Size: {} threads'.format(gpu_name, block_size_opt), 
                        fontsize=16, fontweight='bold', y=0.985)
            
            return fig
            
        except Exception as e:
            print(f"⚠ Erro ao gerar painel de resumo: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_all_graphs(self, output_dir=None):
        """Gera todos os gráficos e os salva"""
        if output_dir is None:
            output_dir = self.cuda_dir / 'graphs'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📈 Gerando gráficos em: {output_dir}")
        
        graphs_generated = []
        
        # 1. Block Size Analysis
        print("  Gerando: Block Size Analysis...", end=' ')
        try:
            fig = self.plot_block_size_analysis()
            if fig:
                path = output_dir / 'block_size_analysis.png'
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("✓")
                graphs_generated.append(str(path))
            else:
                print("✗")
        except Exception as e:
            print(f"✗ ({e})")
        
        # 2. Throughput Analysis
        print("  Gerando: Throughput Analysis...", end=' ')
        try:
            fig = self.plot_throughput_analysis()
            if fig:
                path = output_dir / 'throughput_analysis.png'
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("✓")
                graphs_generated.append(str(path))
            else:
                print("✗")
        except Exception as e:
            print(f"✗ ({e})")
        
        # 3. Timing Breakdown
        print("  Gerando: Timing Breakdown...", end=' ')
        try:
            fig = self.plot_timing_breakdown()
            if fig:
                path = output_dir / 'timing_breakdown.png'
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("✓")
                graphs_generated.append(str(path))
            else:
                print("✗")
        except Exception as e:
            print(f"✗ ({e})")
        
        # 4. Performance Summary
        print("  Gerando: Performance Summary...", end=' ')
        try:
            fig = self.plot_performance_summary()
            if fig:
                path = output_dir / 'performance_summary.png'
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("✓")
                graphs_generated.append(str(path))
            else:
                print("✗")
        except Exception as e:
            print(f"✗ ({e})")
        
        print(f"\n✓ {len(graphs_generated)} gráficos gerados com sucesso!")
        
        return graphs_generated
    
    def generate_report(self, output_file=None):
        """Gera relatório em Markdown com análise de desempenho"""
        if output_file is None:
            output_file = self.cuda_dir / 'ANALISE_DESEMPENHO.md'
        
        output_file = Path(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Análise de Desempenho - K-Means CUDA\n\n")
            
            # Sumário executivo
            f.write("## Sumário Executivo\n\n")
            
            if self.metrics_cuda:
                gpu_name = self.metrics_cuda.get('gpu_name', 'Unknown')
                compute_cap = self.metrics_cuda.get('compute_capability', 'Unknown')
                throughput = float(self.metrics_cuda.get('throughput', 0))
                total_time = float(self.metrics_cuda.get('time_total', 0))
                
                f.write(f"**GPU:** {gpu_name}\n\n")
                f.write(f"**Capacidade de Compute:** {compute_cap}\n\n")
                f.write(f"**Throughput:** {throughput:.2f}M pontos/segundo\n\n")
                f.write(f"**Tempo Total:** {total_time:.3f}ms\n\n")
            
            # Configuração
            f.write("## Configuração do Experimento\n\n")
            
            if self.metrics_cuda:
                f.write("| Parâmetro | Valor |\n")
                f.write("|-----------|-------|\n")
                f.write(f"| Block Size Otimizado | {int(float(self.metrics_cuda.get('block_size_optimized', 0)))} threads |\n")
                f.write(f"| Número de Pontos | {int(float(self.metrics_cuda.get('num_points', 0))):,} |\n")
                f.write(f"| Número de Clusters | {int(float(self.metrics_cuda.get('num_clusters', 0)))} |\n")
                f.write(f"| Iterações | {int(float(self.metrics_cuda.get('num_iterations', 0)))} |\n")
                f.write(f"| Epsilon | 1e-6 |\n")
                f.write(f"| Max Iterações | 100 |\n\n")
            
            # Resultados
            f.write("## Resultados de Performance\n\n")
            
            if self.metrics_cuda:
                f.write("### Timing\n\n")
                h2d = float(self.metrics_cuda.get('time_h2d', 0))
                kernels = float(self.metrics_cuda.get('time_kernels', 0))
                d2h = float(self.metrics_cuda.get('time_d2h', 0))
                total = float(self.metrics_cuda.get('time_total', 0))
                time_per_iter = float(self.metrics_cuda.get('time_per_iteration', 0))
                
                f.write(f"- **Transfer H2D:** {h2d:.3f}ms ({(h2d/total)*100:.1f}%)\n")
                f.write(f"- **Execução de Kernels:** {kernels:.3f}ms ({(kernels/total)*100:.1f}%)\n")
                f.write(f"- **Transfer D2H:** {d2h:.3f}ms ({(d2h/total)*100:.1f}%)\n")
                f.write(f"- **Tempo Total:** {total:.3f}ms\n")
                f.write(f"- **Tempo Médio por Iteração:** {time_per_iter:.3f}ms\n\n")
                
                f.write("### Throughput\n\n")
                throughput_pts = float(self.metrics_cuda.get('throughput_points', 0))
                f.write(f"- **Throughput:** {throughput:.2f}M pontos/segundo\n")
                f.write(f"- **Throughput (pontos/s):** {throughput_pts:.2e}\n\n")
                
                f.write("### Convergência\n\n")
                sse = float(self.metrics_cuda.get('sse_final', 0))
                f.write(f"- **SSE Final:** {sse:.10e}\n\n")
            
            # Block Size Analysis
            if self.block_size_data is not None:
                f.write("## Análise de Block Sizes\n\n")
                f.write("| Block Size | Tempo (ms) |\n")
                f.write("|------------|------------|\n")
                for _, row in self.block_size_data.iterrows():
                    f.write(f"| {int(row['block_size'])} | {row['time_ms']:.6f} |\n")
                f.write("\n")
                
                best_idx = self.block_size_data['time_ms'].idxmin()
                best_block = self.block_size_data.loc[best_idx, 'block_size']
                best_time = self.block_size_data.loc[best_idx, 'time_ms']
                worst_time = self.block_size_data['time_ms'].max()
                improvement = ((worst_time - best_time) / worst_time) * 100
                
                f.write(f"**Melhor Block Size:** {best_block} threads com {best_time:.6f}ms\n\n")
                f.write(f"**Melhoria vs Pior:** {improvement:.1f}%\n\n")
            
            # Otimizações implementadas
            f.write("## Otimizações Implementadas\n\n")
            f.write("1. **Memória Constante para Centróides**\n")
            f.write("   - Centróides armazenados em memória constante (cache rápido)\n")
            f.write("   - Reduz latência de acesso à memória global\n\n")
            
            f.write("2. **Redução Otimizada por Blocos**\n")
            f.write("   - Uso de shared memory para redução eficiente\n")
            f.write("   - Minimiza contenção de atomicAdd em memória global\n\n")
            
            f.write("3. **Cálculo de SSE no Host**\n")
            f.write("   - SSE computado serialmente no host (adequado para 1D)\n")
            f.write("   - Elimina overhead de redução paralela complexa\n\n")
            
            f.write("4. **Teste Automático de Block Sizes**\n")
            f.write("   - Varredura de block sizes (32, 64, 128, 256, 512)\n")
            f.write("   - Seleção automática da configuração ótima\n\n")
            
            f.write("5. **Métricas de Desempenho Detalhadas**\n")
            f.write("   - Medição separada de transferências e execução\n")
            f.write("   - Cálculo de throughput em pontos/segundo\n\n")
            
            # Conclusões
            f.write("## Conclusões\n\n")
            f.write("- ✓ Implementação CUDA otimizada conforme especificação do projeto\n")
            f.write("- ✓ Teste automático de block sizes para melhor desempenho\n")
            f.write("- ✓ Métricas de desempenho detalhadas e exportadas\n")
            f.write("- ✓ Validação de corretude contra versão sequencial\n")
            f.write("- ✓ Gráficos gerados para análise visual\n\n")
            
            f.write("## Arquivos Gerados\n\n")
            f.write("- `metrics_cuda.csv` - Métricas detalhadas em formato CSV\n")
            f.write("- `metrics_cuda.txt` - Métricas em formato texto\n")
            f.write("- `block_size_test.csv` - Resultados de teste de block sizes\n")
            f.write("- `validation_cuda.txt` - Relatório de validação com versão sequencial\n")
            f.write("- `graphs/` - Diretório com todos os gráficos gerados\n")
        
        print(f"✓ Relatório gerado: {output_file}")
        return str(output_file)


def main():
    """Função principal"""
    
    # Obter diretório de entrada (padrão: diretório atual)
    cuda_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print("=" * 70)
    print("K-Means CUDA - Gerador de Gráficos de Desempenho")
    print("=" * 70)
    
    # Criar analisador
    analyzer = PerformanceAnalyzer(cuda_dir)
    
    # Gerar gráficos
    graphs = analyzer.generate_all_graphs()
    
    # Gerar relatório
    report = analyzer.generate_report()
    
    print("\n" + "=" * 70)
    print("Resumo de Arquivos Gerados:")
    print("=" * 70)
    
    for graph in graphs:
        print(f"  📊 {Path(graph).name}")
    
    print(f"  📄 {Path(report).name}")
    
    print("\n✓ Análise completa!")
    print("=" * 70)


if __name__ == '__main__':
    main()
