import os
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

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


    def _expand_y_limits(self, ax, kind='generic'):
        ymin, ymax = ax.get_ylim()
        if ymin == ymax:
            delta = abs(ymax) * 0.1 if ymax != 0 else 1.0
            ax.set_ylim(ymin - delta, ymax + delta)
            return

        span = ymax - ymin
        pad_bottom = 0.03 * span
        pad_top = 0.15 * span

        if kind == 'percent':
            new_top = ymax + max(5.0, 0.12 * max(abs(ymax), 100.0), pad_top)
            new_bottom = ymin - max(1.0, pad_bottom)

            if ymax >= 95:
                new_top = max(new_top, 115.0)
            ax.set_ylim(new_bottom, new_top)
        else:
            new_bottom = ymin - pad_bottom
            new_top = ymax + max(pad_top, 0.08 * max(abs(ymax), 1.0))
            ax.set_ylim(new_bottom, new_top)

    def _add_attack_regions(self, ax, attack_regions, alpha=0.55, show_legend=True, show_arrows=True):
        if not attack_regions:
            return

        added_attack_labels = set()
        for start, end, attack_idx in attack_regions:
            label = self._clean_attack_label(attack_idx)
            color = self.bg_colors[attack_idx % len(self.bg_colors)]
            label_to_show = label if show_legend and label not in added_attack_labels else ''

            ax.axvspan(start, end, facecolor=color, alpha=alpha, zorder=1, label=label_to_show)
            mid = (start + end) / 2
            ax.axvline(mid, color=color, alpha=0.95, linewidth=1.6, zorder=2)

            if show_arrows:
                ax.text(
                    mid,
                    0.89,
                    label,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold',
                    color=color,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
                    clip_on=True,
                    zorder=10,
                )

            if label_to_show:
                added_attack_labels.add(label)

    def plot_score(self, results, attack_regions, title="Análise de Scores", discretization=0.5, scenario_name="General", discretization_strategy="fixed"):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        has_std = False
        algo_name = list(results.keys())[0] if results else "General"

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

        if str(discretization) != 'params':
            ax.axhline(y=discretization, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Threshold ({discretization})', zorder=4)

        self._expand_y_limits(ax, kind='generic')
        self._add_attack_regions(ax, attack_regions, alpha=0.58, show_legend=True, show_arrows=True)

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
            self._add_attack_regions(ax, attack_regions, alpha=0.55, show_legend=True, show_arrows=True)
                    
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
            self._add_attack_regions(ax, attack_regions, alpha=0.55, show_legend=True, show_arrows=True)
                    
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
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
        
        fig.subplots_adjust(bottom=0.15, top=0.94, hspace=0.35) 
        
        output_dir = os.path.join("output", algo_name, discretization_strategy, "Plots", f"{algo_name}_{scenario_name}")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{algo_name}_{title}_FP_FN.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_summary_table(self, csv_dict, algo_name="Algoritmo", include_blocks=None, exclude_blocks=None):
        all_data = []
        
        # Mapeamento fixo de envenenamento (valores sem o símbolo %)
        env_map = {
            'Consistência': {'25': '0,48', '200': '3,34', '1000': '16,08'},
            'Consistencia': {'25': '0,48', '200': '3,34', '1000': '16,08'},
            'Generalização': {'25': '0,49', '200': '3,81', '1000': '16,65'},
            'Generalizacao': {'25': '0,49', '200': '3,81', '1000': '16,65'},
            'Adaptação': {'25': '0,47', '200': '3,79', '1000': '16,63'},
            'Adaptacao': {'25': '0,47', '200': '3,79', '1000': '16,63'},
            'Recorrência': {'25': '0,47', '200': '3,83', '1000': '17,13'},
            'Recorrencia': {'25': '0,47', '200': '3,83', '1000': '17,13'}
        }

        for path, exec_id in csv_dict.items():
            if not os.path.exists(path):
                continue

            filename = os.path.basename(path).lower()
            is_optimized = "otimizado" in filename or "optimized" in filename
            has_33_features = "33features" in filename or "33_features" in filename

            otimizacao = "Otim." if is_optimized else "Def."
            features_qtd = "33" if has_33_features else "77"

            df = pd.read_csv(path, sep=';', decimal=',')

            if exec_id is not None and 'Exec_ID' in df.columns:
                df = df[df['Exec_ID'].astype(str) == str(exec_id)]
                
            if 'Dataset' in df.columns:
                if include_blocks:
                    df = df[df['Dataset'].astype(str).apply(lambda x: any(str(b) in x for b in include_blocks))]
                if exclude_blocks:
                    df = df[~df['Dataset'].astype(str).apply(lambda x: any(str(b) in x for b in exclude_blocks))]

            if df.empty:
                continue

            def get_col(base_name):
                if f"{base_name}_avg" in df.columns: return f"{base_name}_avg"
                if base_name in df.columns: return base_name
                return None

            col_f1 = get_col('F1')
            col_prec = get_col('Prec')
            col_rec = get_col('Rec')
            col_fp = get_col('FP')
            col_fn = get_col('FN')

            df['Cenário'] = df['Dataset'].apply(lambda x: str(x).split('_')[0])
            df['Bloco_Num'] = df['Dataset'].apply(lambda x: int(str(x).split('_')[1]) if '_' in str(x) and str(x).split('_')[1].isdigit() else 0)
            df['Bloco_Str'] = df['Dataset'].apply(lambda x: str(x).split('_')[1] if '_' in str(x) else "-")
            df['Env (%)'] = df.apply(lambda row: env_map.get(row['Cenário'], {}).get(row['Bloco_Str'], row['Bloco_Str']), axis=1)

            cols_to_extract = ['Cenário', 'Env (%)', 'Bloco_Num', col_f1, col_prec, col_rec, col_fp, col_fn]
            df_filtered = df[cols_to_extract].copy()
            df_filtered.columns = ['Cenário', 'Env (%)', 'Bloco_Num', 'F1', 'Prec.', 'Rec.', 'FP', 'FN']

            df_filtered.insert(2, 'Otim.', otimizacao)
            df_filtered.insert(3, 'Feats', features_qtd)

            all_data.append(df_filtered)

        if not all_data:
            return

        df_final = pd.concat(all_data, ignore_index=True)
        
        cenario_order = ['Consistência', 'Consistencia', 'Generalização', 'Generalizacao', 'Adaptação', 'Adaptacao', 'Recorrência', 'Recorrencia']
        df_final['Cenario_Cat'] = pd.Categorical(df_final['Cenário'], categories=cenario_order, ordered=True)
        df_final = df_final.sort_values(by=['Cenario_Cat', 'Bloco_Num', 'Otim.', 'Feats']).reset_index(drop=True)

        numeric_cols = ['F1', 'Prec.', 'Rec.', 'FP', 'FN']
        for col in numeric_cols:
            df_final[col] = pd.to_numeric(df_final[col].astype(str).str.replace(',', '.'), errors='coerce')

        # Mapeamento para negrito (vencedores por Cenário E Nível de Envenenamento)
        best_indices_map = {}
        for scenario in df_final['Cenário'].unique():
            scenario_df = df_final[df_final['Cenário'] == scenario]
            for env in scenario_df['Env (%)'].unique():
                group_df = scenario_df[scenario_df['Env (%)'] == env]
                if not group_df.empty:
                    for col in ['F1', 'Prec.', 'Rec.']:
                        max_val = group_df[col].max()
                        if pd.notnull(max_val):
                            best_indices_map.setdefault(col, []).extend(group_df[group_df[col] == max_val].index.tolist())
                    for col in ['FP', 'FN']:
                        min_val = group_df[col].min()
                        if pd.notnull(min_val):
                            best_indices_map.setdefault(col, []).extend(group_df[group_df[col] == min_val].index.tolist())

        df_final_orig = df_final.copy()
        df_final.drop(columns=['Bloco_Num', 'Cenario_Cat'], inplace=True)

        # Preparação do DataFrame para LaTeX
        df_latex = df_final.copy()
        for col in ['F1', 'Prec.', 'Rec.']:
            df_latex[col] = df_latex.apply(lambda row: f"\\textbf{{{row[col]:.2f}}}" if row.name in best_indices_map.get(col, []) else f"{row[col]:.2f}", axis=1).str.replace('.', ',')
        for col in ['FP', 'FN']:
            df_latex[col] = df_latex.apply(lambda row: f"\\textbf{{{int(row[col])}}}" if row.name in best_indices_map.get(col, []) else str(int(row[col])), axis=1)

        # Formatação para PNG no Jupyter
        for col in ['F1', 'Prec.', 'Rec.']:
            df_final[col] = df_final[col].apply(lambda x: f"{x:.2f}".replace('.', ',') if pd.notnull(x) else "-")
        for col in ['FP', 'FN']:
            df_final[col] = df_final[col].apply(lambda x: str(int(x)) if pd.notnull(x) else "-")

        display_values = df_final.values.copy()
        for i in range(len(display_values) - 1, 0, -1):
            if display_values[i, 0] == display_values[i-1, 0]:
                display_values[i, 0] = ""

        # Geração da Tabela 
        col_widths = [0.18, 0.10, 0.08, 0.08, 0.12, 0.12, 0.12, 0.10, 0.10]
        fig, ax = plt.subplots(figsize=(10, 0.4 * len(df_final) + 0.5))
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.05, right=0.95)
        ax.axis('off')
        
        table = ax.table(cellText=display_values, colLabels=df_final.columns, loc='center', cellLoc='center', colWidths=col_widths)
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.8)

        for (row, col_idx), cell in table.get_celld().items():
            cell.set_facecolor('white'); cell.set_edgecolor('black'); cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(weight='bold')
            elif row > 0:
                col_name = df_final.columns[col_idx]
                if (row-1) in best_indices_map.get(col_name, []):
                    cell.set_text_props(weight='bold')

        plt.title(f"Resultados - {algo_name}", fontsize=14, fontweight='bold', pad=0)
        plt.show()

        # Geração de Saída LaTeX
        print("\n" + "="*65 + "\nCÓDIGO LATEX PARA OVERLEAF:\n" + "="*65)
        print("\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\n\\hline")
        headers_tex = ["Cenário", "Env (\\%)", "Otim.", "Feats", "F1", "Prec.", "Rec.", "FP", "FN"]
        print(" & ".join([f"\\textbf{{{h}}}" for h in headers_tex]) + " \\\\ \\hline")
        
        scenario_counts = df_final_orig['Cenário'].value_counts(sort=False).to_dict()
        last_s = None
        
        for i in range(len(df_latex)):
            curr_s = df_final_orig.iloc[i]['Cenário']
            row_cells = list(df_latex.iloc[i].values)
            
            if curr_s != last_s:
                if last_s is not None: print("\\hline")
                count = scenario_counts[curr_s]
                row_cells[0] = f"\\multirow{{{count}}}{{*}}{{{curr_s}}}"
                last_s = curr_s
            else:
                row_cells[0] = ""
            
            print(" & ".join(row_cells) + " \\\\")
        
        print(f"\\hline\n\\end{{tabular}}\n\\caption{{Resultados do algoritmo {algo_name}}}\n\\end{{table}}\n" + "="*65 + "\n")