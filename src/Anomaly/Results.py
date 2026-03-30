import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
import warnings
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")

class Metrics:
    def get_metric_classifier(self, metrics_dict, metric_name, target_class=None):
        norm_metrics = {str(k).lower(): v for k, v in metrics_dict.items()}
        metric_name = str(metric_name).lower()
        
        if target_class is None or str(target_class).lower() == 'macro':
            tc = 'macro'
        else:
            tc = int(target_class)

        if tc == 'macro':
            val_0 = norm_metrics.get(f'{metric_name}_0', 0.0)
            val_1 = norm_metrics.get(f'{metric_name}_1', 0.0)
            val_0 = 0.0 if val_0 is None or np.isnan(val_0) else float(val_0)
            val_1 = 0.0 if val_1 is None or np.isnan(val_1) else float(val_1)
            return (val_0 + val_1) / 2.0
        else:
            val = norm_metrics.get(f'{metric_name}_{tc}')
            return float(val) if val is not None and not np.isnan(val) else 0.0

    def calc_sklearn_metrics(self, y_true, y_pred, target_class=None):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        if target_class is None or str(target_class).lower() == 'macro':
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                fpr = 0.0
                tpr = 0.0
        else:
            tc = int(target_class)
            f1 = f1_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                if tc == 0:
                    fpr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    tpr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                fpr = 0.0
                tpr = 0.0
            
        return f1 * 100.0, prec * 100.0, rec * 100.0, mcc, fpr * 100.0, tpr * 100.0

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
            
            # F1 Inicio (antes da onda de ataque) - Usa target_class_pass
            y_t_start = y_true[warmup_instances:start_eff]
            y_p_start = y_pred[warmup_instances:start_eff]
            f1_start = self.calc_sklearn_metrics(y_t_start, y_p_start, target_class_pass)[0] if len(y_t_start) > 0 else 0.0
            
            # F1 Final (ao final da onda de ataque) - Usa target_class_pass
            y_t_end = y_true[warmup_instances:end+1]
            y_p_end = y_pred[warmup_instances:end+1]
            f1_end = self.calc_sklearn_metrics(y_t_end, y_p_end, target_class_pass)[0] if len(y_t_end) > 0 else 0.0
            
            # F1 Recuperação (após X amostras do fim da onda) - Usa target_class_pass
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

        header_base = f"{'Algoritmo':<22} | {'F1 (%)':<8} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'MCC':<8} | {'Tempo (s)':<10}"
        line_len = len(header_base)
        
        # Strings para o cabeçalho
        if target_class is None:
            gen_str = "HÍBRIDA (Macro Global)"
        elif str(target_class).lower() == 'macro':
            gen_str = "MACRO"
        else:
            gen_str = f"CLASSE {target_class}"

        if tc_pass is None or str(tc_pass).lower() == 'macro':
            beh_str = "MACRO"
        else:
            beh_str = f"CLASSE {tc_pass}"

        # Monta um título
        titulo_relatorio = f"RELATÓRIO COMPORTAMENTAL | Métricas: {gen_str} | Passagens: {beh_str}"
        line_len = max(len(header_base), len(titulo_relatorio) + 4)

        print(f"\n{'='*line_len}")
        print(f"{titulo_relatorio:^{line_len}}")
        print(f"{'='*line_len}")
        print(header_base)
        print(f"{'-'*line_len}")

        for name, data in predictions_history.items():
            y_true_full = np.array(data['true_labels'])
            y_pred_full = np.array(data['predicted_classes'])
            
            y_true_list = y_true_full[warmup_instances:] if len(y_true_full) > warmup_instances else y_true_full
            y_pred_list = y_pred_full[warmup_instances:] if len(y_pred_full) > warmup_instances else y_pred_full
            
            f1, prec, recall, mcc, fpr, tpr = self.calc_sklearn_metrics(y_true_list, y_pred_list, target_class)
            exec_time = data.get('exec_time', 0.0)

            row_base = f"{name:<22} | {f1:<8.2f} | {prec:<8.2f} | {recall:<8.2f} | {mcc:<8.3f} | {exec_time:<10.2f}"
            print(row_base)
            
            # Printa as métricas de cada onda de ataque
            behavioral_data = self.calc_behavioral_metrics(y_true_full, y_pred_full, attack_regions, recovery_window, warmup_instances, tc_pass)
            print(f"{'-'*line_len}")
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
        self.target_names = target_names
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        self.bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']

    def plot_score(self, results, attack_regions, title, threshold=0.5):
        fig, ax = plt.subplots(figsize=(15, 6)) 
        
        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            scores = np.array(data['scores'])
            instances = np.arange(len(scores))
            window_size = 5
            
            moving_avg = np.array([np.mean(scores[max(0, j-window_size):j+1]) for j in range(len(scores))])
            ax.plot(instances, moving_avg, color=color, alpha=0.85, linewidth=1.5, label=f'{name}', zorder=3)

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Threshold ({threshold})', zorder=4)

        added_attack_labels = set()
        for start, end, attack_idx in attack_regions:
            attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
            bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
            
            label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
            ax.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
            
            if label_to_show:
                added_attack_labels.add(attack_name)

        ax.set_title(f"Análise de Scores (Média Móvel - Janela {window_size})", fontsize=14, fontweight='bold')
        ax.set_ylabel("Score de Anomalia", fontsize=14)
        ax.set_xlabel("Instâncias", fontsize=14)
        
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(results) + 2, 
                  fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)

        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
        fig.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.show()

    def plot_metrics(self, results, attack_regions, title, window_size, target_class=None):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        for i, (name, data) in enumerate(results.items()):
            color = self.colors[i % len(self.colors)]
            ax1.plot(data['instances'], data['f1_score'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
            ax2.plot(data['instances'], data['precision'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)
            ax3.plot(data['instances'], data['recall'], label=f'{name}', color=color, linewidth=2.5, zorder=3, marker='o', markersize=5)

        for ax in [ax1, ax2, ax3]:
            added_attack_labels = set()
            for start, end, attack_idx in attack_regions:
                attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
                
                label_to_show = f'{attack_name}' if attack_name not in added_attack_labels else ""
                ax.axvspan(start, end, facecolor=bg_color, alpha=0.4, zorder=2, label=label_to_show)
                
                if label_to_show:
                    added_attack_labels.add(attack_name)
                    
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
        if target_class is None or str(target_class).lower() == 'macro':
            tgt_str = "Macro Global"
        elif str(target_class) == '0':
            tgt_str = "Classe 0 (Normal)"
        elif str(target_class) == '1':
            tgt_str = "Classe 1 (Ataque)"
        else:
            tgt_str = f"Classe {target_class}"
        
        ax1.set_title(f"{title} - Evolução {tgt_str} (Resolução de {window_size} instâncias)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("F1-Score (%)", fontsize=14)
        ax2.set_ylabel("Precision (%)", fontsize=14)
        ax3.set_ylabel("Recall (%)", fontsize=14)
        ax3.set_xlabel("Instâncias", fontsize=14)

        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=len(results) + 2, 
                  fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)
        
        fig.subplots_adjust(bottom=0.3)
        plt.tight_layout()
        plt.show()