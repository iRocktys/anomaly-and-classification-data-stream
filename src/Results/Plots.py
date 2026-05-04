import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import os
import pandas as pd

class Plots:
    def __init__(self, target_names):
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        self.bg_colors = ['#ff4d4d', '#4d88ff', '#2ecc71', '#ffb84d', '#b366ff', '#ff66b3', '#33cccc']


    def _clean_attack_label(self, attack_idx):
        if attack_idx < len(self.target_names):
            label = str(self.target_names[attack_idx])
        else:
            label = f'Classe {attack_idx}'

        label = re.sub(r'(?i)^drdos[_\-\s]*', '', label)
        label = re.sub(r'(?i)^ddos[_\-\s]*', '', label)
        label = label.replace('_', ' ').strip()
        return label if label else f'Classe {attack_idx}'

    def _moving_average(self, values, window_size):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        if window_size is None or window_size <= 1:
            return values.copy()

        window_size = min(int(window_size), len(values))
        kernel = np.ones(window_size, dtype=float) / float(window_size)
        valid = np.convolve(values, kernel, mode='valid')
        prefix = [np.mean(values[:i + 1]) for i in range(window_size - 1)]
        return np.concatenate([np.array(prefix, dtype=float), valid])

    def _expand_y_limits(self, ax, kind='generic'):
        ymin, ymax = ax.get_ylim()
        if ymin == ymax:
            delta = abs(ymax) * 0.1 if ymax != 0 else 1.0
            ax.set_ylim(ymin - delta, ymax + delta)
            return

        span = ymax - ymin
        pad_bottom = 0.03 * span

        if kind == 'percent':
            new_top = ymax + max(5.0, 0.12 * max(abs(ymax), 100.0), 0.15 * span)
            new_bottom = ymin - max(1.0, pad_bottom)
            if ymax >= 95:
                new_top = max(new_top, 115.0)
            ax.set_ylim(new_bottom, new_top)
        else:
            new_bottom = ymin - pad_bottom
            new_top = ymax + max(0.15 * span, 0.08 * max(abs(ymax), 1.0))
            ax.set_ylim(new_bottom, new_top)

    def _add_attack_regions(self, ax, attack_regions, alpha=0.55, show_legend=True, show_labels=True):
        if not attack_regions:
            return

        added_attack_labels = set()
        for start, end, attack_idx in attack_regions:
            attack_name = self._clean_attack_label(attack_idx)
            bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
            label_to_show = attack_name if show_legend and attack_name not in added_attack_labels else ''

            ax.axvspan(start, end, facecolor=bg_color, alpha=alpha, zorder=1, label=label_to_show)
            mid = (start + end) / 2
            ax.axvline(mid, color=bg_color, alpha=0.95, linewidth=1.6, zorder=2)

            if show_labels:
                ax.text(
                    mid,
                    0.89,
                    attack_name,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold',
                    color=bg_color,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
                    clip_on=True,
                    zorder=10,
                )

            if label_to_show:
                added_attack_labels.add(attack_name)

    def plot_score(self, results, attack_regions, title="Análise de Scores", discretization=0.5, scenario_name="General", discretization_strategy="fixed"):
        fig, ax = plt.subplots(figsize=(15, 6))

        has_std = False
        algo_name = list(results.keys())[0] if results else "General"

        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]

            if 'scores_mean' in data:
                scores_raw = np.array(data['scores_mean'], dtype=float)
                scores_std = np.array(data.get('scores_std', np.zeros_like(scores_raw)), dtype=float)
                instances = np.arange(len(scores_raw))
            elif 'scores' in data:
                scores_raw = np.array(data['scores'], dtype=float)
                scores_std = None
                instances = np.arange(len(scores_raw))
            else:
                continue

            trend_window = max(5, min(25, len(scores_raw) // 50 if len(scores_raw) >= 50 else 5))
            trend_scores = self._moving_average(scores_raw, trend_window)

            ax.plot(
                instances,
                scores_raw,
                color=color,
                alpha=0.18,
                linewidth=0.8,
                zorder=2,
            )

            ax.plot(
                instances,
                trend_scores,
                color=color,
                alpha=0.95,
                linewidth=2.2,
                label=f'{name}',
                zorder=4,
            )

            if scores_std is not None and np.sum(scores_std) > 0:
                trend_std = self._moving_average(scores_std, trend_window)
                ax.fill_between(
                    instances,
                    trend_scores - trend_std,
                    trend_scores + trend_std,
                    color='gray',
                    alpha=0.22,
                    zorder=3,
                )
                has_std = True

        if str(discretization) != 'params':
            ax.axhline(y=discretization, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Threshold ({discretization})', zorder=5)

        self._expand_y_limits(ax, kind='generic')
        self._add_attack_regions(ax, attack_regions, alpha=0.58, show_legend=True, show_labels=True)

        ax.set_title(f"{algo_name} - {title}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score de Anomalia", fontsize=14)
        ax.set_xlabel("Instâncias", fontsize=14)

        handles, labels = ax.get_legend_handles_labels()
        if has_std and 'Desvio Padrão' not in labels:
            handles.append(mpatches.Patch(color='gray', alpha=0.3, label='Desvio Padrão'))
            labels.append('Desvio Padrão')

        leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(results) + 2, fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)

        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
        fig.subplots_adjust(bottom=0.2, top=0.94)

        output_dir = os.path.join("output", algo_name, discretization_strategy, "Plots", f"{algo_name}_{scenario_name}")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{algo_name}_{title}_Scores.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_metrics(self, results, attack_regions=None, title="Métricas", window_size=1000, target_class=None, scenario_name="General", discretization_strategy="fixed"):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        has_std = False
        algo_name = list(results.keys())[0] if results else "General"
        def clean(d_list): return np.array([0.0 if (v is None or np.isnan(v)) else v for v in d_list])

        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            x_axis = data['instances']
            
            if 'f1_mean' in data:
                f1_m, f1_s = clean(data['f1_mean']), clean(data['f1_std'])
                pr_m, pr_s = clean(data['precision_mean']), clean(data['precision_std'])
                re_m, re_s = clean(data['recall_mean']), clean(data['recall_std'])
                
                ax1.plot(x_axis, f1_m, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                ax2.plot(x_axis, pr_m, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                ax3.plot(x_axis, re_m, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                
                if np.sum(f1_s) > 0:
                    ax1.fill_between(x_axis, f1_m - f1_s, f1_m + f1_s, color='gray', alpha=0.3, zorder=2)
                    ax2.fill_between(x_axis, pr_m - pr_s, pr_m + pr_s, color='gray', alpha=0.3, zorder=2)
                    ax3.fill_between(x_axis, re_m - re_s, re_m + re_s, color='gray', alpha=0.3, zorder=2)
                    has_std = True
            
            else:
                f1_data = clean(data.get('f1', data.get('f1_score', [])))
                ax1.plot(x_axis, f1_data, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                ax2.plot(x_axis, clean(data['precision']), label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                ax3.plot(x_axis, clean(data['recall']), label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)

        for ax in [ax1, ax2, ax3]:
            self._expand_y_limits(ax, kind='percent')
            self._add_attack_regions(ax, attack_regions, alpha=0.55, show_legend=True, show_labels=True)
                    
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
        ax1.set_title(f"{algo_name} - {title} (Métricas por janela de {window_size} instâncias)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("F1-Score por janela (%)", fontsize=14)
        ax2.set_ylabel("Precision por janela (%)", fontsize=14)
        ax3.set_ylabel("Recall por janela (%)", fontsize=14)
        ax3.set_xlabel("Instâncias", fontsize=14)

        handles, labels = ax1.get_legend_handles_labels()
        
        if has_std and 'Desvio Padrão' not in labels:
            handles.append(mpatches.Patch(color='gray', alpha=0.3, label='Desvio Padrão'))
            labels.append('Desvio Padrão')
            
        leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(results) + 2, fontsize=12, frameon=False)
        for patch in leg.get_patches(): 
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)
        
        fig.subplots_adjust(bottom=0.15, top=0.94, hspace=0.35) 
        
        output_dir = os.path.join("output", algo_name, discretization_strategy, "Plots", f"{algo_name}_{scenario_name}")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{algo_name}_{title}_Metricas.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_fp_fn(self, results, attack_regions=None, title="Contagem de FP e FN", window_size=1000, scenario_name="General", discretization_strategy="fixed"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        
        has_std = False
        algo_name = list(results.keys())[0] if results else "General"
        def clean(d_list): return np.array([0.0 if (v is None or np.isnan(v)) else v for v in d_list])

        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            x_axis = data['instances']
            
            if 'fp_mean' in data:
                fp_m = np.ceil(clean(data['fp_mean']))
                fp_s = np.ceil(clean(data['fp_std']))
                fn_m = np.ceil(clean(data['fn_mean']))
                fn_s = np.ceil(clean(data['fn_std']))
                
                ax1.plot(x_axis, fp_m, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                ax2.plot(x_axis, fn_m, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                
                if np.sum(fp_s) > 0 or np.sum(fn_s) > 0:
                    ax1.fill_between(x_axis, fp_m - fp_s, fp_m + fp_s, color='gray', alpha=0.3, zorder=2)
                    ax2.fill_between(x_axis, fn_m - fn_s, fn_m + fn_s, color='gray', alpha=0.3, zorder=2)
                    has_std = True
            
            else:
                fp_data = np.ceil(clean(data.get('fp', [])))
                fn_data = np.ceil(clean(data.get('fn', [])))
                
                ax1.plot(x_axis, fp_data, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                ax2.plot(x_axis, fn_data, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)

        for ax in [ax1, ax2]:
            self._expand_y_limits(ax, kind='generic')
            self._add_attack_regions(ax, attack_regions, alpha=0.55, show_legend=True, show_labels=True)
                    
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
        ax1.set_title(f"{algo_name} - {title} (FP/FN por janela de {window_size} instâncias)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Falsos Positivos (FP)", fontsize=14)
        ax2.set_ylabel("Falsos Negativos (FN)", fontsize=14)
        ax2.set_xlabel("Instâncias", fontsize=14)

        handles, labels = ax1.get_legend_handles_labels()
        
        if has_std and 'Desvio Padrão' not in labels:
            handles.append(mpatches.Patch(color='gray', alpha=0.3, label='Desvio Padrão'))
            labels.append('Desvio Padrão')
            
        leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(results) + 2, fontsize=12, frameon=False)
        for patch in leg.get_patches(): 
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)
        
        fig.subplots_adjust(bottom=0.15, top=0.94, hspace=0.35) 
        
        output_dir = os.path.join("output", algo_name, discretization_strategy, "Plots", f"{algo_name}_{scenario_name}")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{algo_name}_{title}_FP_FN.png"), bbox_inches='tight')
        plt.close(fig)
