# Results.py
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")

class Metrics:
    def calc_sklearn_metrics(self, y_true, y_pred, target_class=1):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0

        if target_class is None or str(target_class).lower() == 'macro':
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            tc = int(target_class)
            f1 = f1_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            
        return f1 * 100.0, prec * 100.0, rec * 100.0

    def extract_attack_regions(self, y_true_multi, normal_class_idx=0):
        y_true_array = np.array(y_true_multi)
        attack_indices = np.where(y_true_array != normal_class_idx)[0]
        
        attack_regions = []
        if len(attack_indices) > 0:
            start_idx = attack_indices[0]
            last_idx = attack_indices[0]
            for idx in attack_indices[1:]:
                if idx - last_idx > 1000:
                    block_labels = y_true_array[start_idx:last_idx+1]
                    block_attack_labels = block_labels[block_labels != normal_class_idx]
                    block_label = np.bincount(block_attack_labels).argmax() if len(block_attack_labels) > 0 else 1
                    attack_regions.append((start_idx, last_idx, block_label))
                    start_idx = idx
                last_idx = idx
            
            block_labels = y_true_array[start_idx:last_idx+1]
            block_attack_labels = block_labels[block_labels != normal_class_idx]
            block_label = np.bincount(block_attack_labels).argmax() if len(block_attack_labels) > 0 else 1
            attack_regions.append((start_idx, last_idx, block_label))
            
        return attack_regions

    def calc_behavioral_metrics(self, y_true, y_pred, attack_regions, recovery_window=1000, warmup_instances=0, target_class_pass=None):
        behavioral_data = []
        for i, (start, end, label) in enumerate(attack_regions):
            if end < warmup_instances:
                behavioral_data.append({'ataque_idx': i + 1, 'passagem': 0.0, 'recuperacao': 0.0})
                continue
                
            start_eff = max(start, warmup_instances)
            
            y_t_start = y_true[warmup_instances:start_eff]
            y_p_start = y_pred[warmup_instances:start_eff]
            f1_start = self.calc_sklearn_metrics(y_t_start, y_p_start, target_class_pass)[0] if len(y_t_start) > 0 else 0.0
            
            y_t_end = y_true[warmup_instances:end+1]
            y_p_end = y_pred[warmup_instances:end+1]
            f1_end = self.calc_sklearn_metrics(y_t_end, y_p_end, target_class_pass)[0] if len(y_t_end) > 0 else 0.0
            
            rec_idx = min(end + 1 + recovery_window, len(y_true))
            y_t_rec = y_true[warmup_instances:rec_idx]
            y_p_rec = y_pred[warmup_instances:rec_idx]
            f1_rec = self.calc_sklearn_metrics(y_t_rec, y_p_rec, target_class_pass)[0] if len(y_t_rec) > 0 else 0.0
            
            passagem = f1_end - f1_start
            recuperacao = f1_rec - f1_end
            
            behavioral_data.append({
                'ataque_idx': i + 1,
                'passagem': passagem,
                'recuperacao': recuperacao
            })
            
        return behavioral_data

    def display_cumulative_metrics(self, predictions_history, warmup_instances=0, target_class=None, target_class_pass=None, attack_regions=None, recovery_window=1000, normal_class_idx=0):
        if attack_regions is None or len(attack_regions) == 0:
            first_name = list(predictions_history.keys())[0]
            y_true_multi = predictions_history[first_name].get('true_labels_multi')
            if y_true_multi is not None:
                attack_regions = self.extract_attack_regions(y_true_multi, normal_class_idx)
            else:
                attack_regions = []

        tc_pass = target_class_pass if target_class_pass is not None else target_class

        if target_class is None:
            gen_str = "HÍBRIDA (Macro Global)"
        elif str(target_class).lower() == 'macro':
            gen_str = "MACRO TOTAL"
        else:
            gen_str = f"CLASSE {target_class}"

        if tc_pass is None or str(tc_pass).lower() == 'macro':
            beh_str = "MACRO"
        else:
            beh_str = f"CLASSE {tc_pass}"

        titulo_relatorio = f"RELATÓRIO COMPORTAMENTAL | Geral: {gen_str} | Comportamento: {beh_str}"
        header_base = f"{'Modelo':<25} | {'F1 (%)':<8} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'Tempo (s)':<10}"
        line_len = max(len(header_base), len(titulo_relatorio) + 4)

        print(f"\n{'='*line_len}")
        print(f"{titulo_relatorio:^{line_len}}")
        print(f"{'='*line_len}")
        print(header_base)
        print(f"{'-'*line_len}")

        for name, data in predictions_history.items():
            y_true_full = np.array(data['y_true'])
            y_pred_full = np.array(data['y_pred'])
            
            y_true_list = y_true_full[warmup_instances:] if len(y_true_full) > warmup_instances else y_true_full
            y_pred_list = y_pred_full[warmup_instances:] if len(y_pred_full) > warmup_instances else y_pred_full
            
            f1, prec, recall = self.calc_sklearn_metrics(y_true_list, y_pred_list, target_class)
            exec_time = data.get('exec_time', 0.0)

            row_base = f"{name:<25} | {f1:<8.2f} | {prec:<8.2f} | {recall:<8.2f} | {exec_time:<10.2f}"
            print(row_base)
            
            behavioral_data = self.calc_behavioral_metrics(y_true_full, y_pred_full, attack_regions, recovery_window, warmup_instances, tc_pass)
            for b in behavioral_data:
                idx = b['ataque_idx']
                p = b['passagem']
                r = b['recuperacao']
                p_str = f"+{p:.2f}%" if p > 0 else f"{p:.2f}%"
                r_str = f"+{r:.2f}%" if r > 0 else f"{r:.2f}%"
                
                print(f"  -> Ataque {idx} ({beh_str}): Passagem: {p_str:<8} | Recuperação ({recovery_window} amostras): {r_str}")
        
        print(f"{'='*line_len}\n")


class Plots:
    def __init__(self, target_names):
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        self.bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']

    def plot_metrics(self, results, attack_regions=None, title="Métricas", window_size=1000):
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        (ax1, ax2, ax3) = axes

        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            x_axis = data['instances']
            
            def clean(d_list): return [0.0 if (v is None or np.isnan(v)) else v for v in d_list]

            ax1.plot(x_axis, clean(data['precision']), label=name, color=color, alpha=0.85, linewidth=1.5, marker='o', zorder=3)
            ax2.plot(x_axis, clean(data['recall']), label=name, color=color, alpha=0.85, linewidth=1.5, marker='o', zorder=3)
            ax3.plot(x_axis, clean(data['f1']), label=name, color=color, alpha=0.85, linewidth=1.5, marker='o', zorder=3)

        added_attack_labels = set()
        for ax in axes:
            if attack_regions:
                for start, end, attack_idx in attack_regions:
                    attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                    bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
                    
                    label_to_show = f'{attack_name}' if attack_name not in added_attack_labels and ax == ax1 else ""
                    ax.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
                    
                    if label_to_show:
                        added_attack_labels.add(attack_name)
            
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)

        ax1.set_title(f"{title} (Resolução de {window_size} instâncias)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Precision (%)", fontsize=12)
        ax2.set_ylabel("Recall (%)", fontsize=12)
        ax3.set_ylabel("F1-Score (%)", fontsize=12)
        ax3.set_xlabel("Instâncias", fontsize=14)

        handles, labels = ax1.get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(results) + len(added_attack_labels), 
                         fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)

        fig.subplots_adjust(bottom=0.15)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()