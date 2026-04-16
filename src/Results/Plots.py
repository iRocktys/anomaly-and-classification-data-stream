import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

class Plots:
    def __init__(self, target_names):
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        self.bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']

    def plot_score(self, results, attack_regions, title="Análise de Scores", discretization=0.5):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        has_std = False
        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            window_size = 2
            
            if 'scores_mean' in data:
                scores_m = np.array(data['scores_mean'])
                scores_s = np.array(data['scores_std'])
                instances = np.arange(len(scores_m))
                
                moving_avg = np.array([np.mean(scores_m[max(0, j-window_size):j+1]) for j in range(len(scores_m))])
                moving_std = np.array([np.mean(scores_s[max(0, j-window_size):j+1]) for j in range(len(scores_s))])
                
                ax.plot(instances, moving_avg, color=color, alpha=0.85, linewidth=1.5, label=f'{name}', zorder=3)
                
                if np.sum(moving_std) > 0:
                    ax.fill_between(instances, moving_avg - moving_std, moving_avg + moving_std, color='gray', alpha=0.3, zorder=2)
                    has_std = True
            
            elif 'scores' in data:
                scores = np.array(data['scores'])
                instances = np.arange(len(scores))
                moving_avg = np.array([np.mean(scores[max(0, j-window_size):j+1]) for j in range(len(scores))])
                ax.plot(instances, moving_avg, color=color, alpha=0.85, linewidth=1.5, label=f'{name}', zorder=3)

        ax.axhline(y=discretization, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'discretization ({discretization})', zorder=4)

        added_attack_labels = set()
        for start, end, attack_idx in attack_regions:
            attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
            bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
            label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
            ax.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
            if label_to_show: added_attack_labels.add(attack_name)

        ax.set_title(f"{title} (Média Móvel - Janela {window_size})", fontsize=14, fontweight='bold')
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
        fig.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.show()

    def plot_metrics(self, results, attack_regions=None, title="Métricas", window_size=1000, target_class=None):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        has_std = False
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
            added_attack_labels = set()
            if attack_regions:
                for start, end, attack_idx in attack_regions:
                    attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                    bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
                    
                    label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
                    ax.axvspan(start, end, facecolor=bg_color, alpha=0.4, zorder=1, label=label_to_show)
                    if label_to_show: added_attack_labels.add(attack_name)
                    
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
        tgt_str = "Macro Global" if target_class is None or str(target_class).lower() == 'macro' else f"Classe {target_class}"
        
        ax1.set_title(f"{title} - Evolução {tgt_str} (Resolução de {window_size} instâncias)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("F1-Score (%)", fontsize=14)
        ax2.set_ylabel("Precision (%)", fontsize=14)
        ax3.set_ylabel("Recall (%)", fontsize=14)
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
        
        fig.subplots_adjust(bottom=0.15, hspace=0.3) 
        plt.show()

    def plot_fp_fn(self, results, attack_regions=None, title="Contagem de FP e FN", window_size=1000):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        
        has_std = False
        def clean(d_list): return np.array([0.0 if (v is None or np.isnan(v)) else v for v in d_list])

        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            x_axis = data['instances']
            
            if 'fp_mean' in data:
                # Aplicando np.ceil aos arrays para forçar números inteiros
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
                # O mesmo tratamento para execuções únicas
                fp_data = np.ceil(clean(data.get('fp', [])))
                fn_data = np.ceil(clean(data.get('fn', [])))
                
                ax1.plot(x_axis, fp_data, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
                ax2.plot(x_axis, fn_data, label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)

        for ax in [ax1, ax2]:
            added_attack_labels = set()
            if attack_regions:
                for start, end, attack_idx in attack_regions:
                    attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                    bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
                    
                    label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
                    ax.axvspan(start, end, facecolor=bg_color, alpha=0.4, zorder=1, label=label_to_show)
                    if label_to_show: added_attack_labels.add(attack_name)
                    
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Força o eixo Y a usar apenas números inteiros
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
        ax1.set_title(f"{title} (Resolução de {window_size} instâncias)", fontsize=14, fontweight='bold')
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
        
        fig.subplots_adjust(bottom=0.15, hspace=0.3) 
        plt.show()