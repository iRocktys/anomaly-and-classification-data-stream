import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
import warnings

warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")

class Metrics:
    def get_metric_classifier(self, metrics_dict, metric_name, target_class=None):
        norm_metrics = {str(k).lower(): v for k, v in metrics_dict.items()}
        metric_name = str(metric_name).lower()
        
        tc = 'macro' if target_class is None or str(target_class).lower() == 'macro' else int(target_class)

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
            f1 = f1_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=1)
            prec = precision_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=1)
            rec = recall_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=1)
            
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
            f1 = f1_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=1)
            prec = precision_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=1)
            rec = recall_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=1)
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

    def display_cumulative_metrics(self, predictions_history, warmup_instances=0, target_class=None, n_runs=1):
        gen_str = "MACRO GLOBAL" if target_class is None or str(target_class).lower() == 'macro' else f"CLASSE {target_class}"

        titulo_relatorio = f"RELATÓRIO DE MÉTRICAS | {gen_str}"
        if n_runs > 1:
            header_base = f"{'Modelo/Algoritmo':<22} | {'F1 (%)':<13} | {'Prec (%)':<13} | {'Rec (%)':<13} | {'MCC':<11} | {'FPR (%)':<13} | {'TPR (%)':<13} | {'Tempo (s)':<12}"
        else:
            header_base = f"{'Modelo/Algoritmo':<22} | {'F1 (%)':<8} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'MCC':<8} | {'FPR (%)':<8} | {'TPR (%)':<8} | {'Tempo (s)':<10}"
            
        line_len = max(len(header_base), len(titulo_relatorio) + 4)

        print(f"\n{'='*line_len}")
        print(f"{titulo_relatorio:^{line_len}}")
        print(f"{'='*line_len}")
        print(header_base)
        print(f"{'-'*line_len}")

        # Armazena linhas formatadas com TAB para o Google Sheets/Excel
        tsv_lines = ["Modelo/Algoritmo\tF1 (%)\tPrec (%)\tRec (%)\tMCC\tFPR (%)\tTPR (%)\tTempo (s)"]

        for name, data in predictions_history.items():
            if 'cumulative' in data:
                f1_m, f1_s = data['cumulative']['f1']
                prec_m, prec_s = data['cumulative']['prec']
                rec_m, rec_s = data['cumulative']['rec']
                mcc_m, mcc_s = data['cumulative']['mcc']
                fpr_m, fpr_s = data['cumulative']['fpr']
                tpr_m, tpr_s = data['cumulative']['tpr']
                tm_m, tm_s = data['exec_time_mean'], data['exec_time_std']
                
                print(f"{'-'*line_len}")
                if n_runs > 1:
                    row_base = f"{name:<22} | {f1_m:>5.2f} ± {f1_s:<5.2f} | {prec_m:>5.2f} ± {prec_s:<5.2f} | {rec_m:>5.2f} ± {rec_s:<5.2f} | {mcc_m:>4.2f} ± {mcc_s:<4.2f} | {fpr_m:>5.2f} ± {fpr_s:<5.2f} | {tpr_m:>5.2f} ± {tpr_s:<5.2f} | {tm_m:>5.2f} ± {tm_s:<4.2f}"
                    tsv_row = f"{name}\t{f1_m:.2f} ± {f1_s:.2f}\t{prec_m:.2f} ± {prec_s:.2f}\t{rec_m:.2f} ± {rec_s:.2f}\t{mcc_m:.2f} ± {mcc_s:.2f}\t{fpr_m:.2f} ± {fpr_s:.2f}\t{tpr_m:.2f} ± {tpr_s:.2f}\t{tm_m:.2f} ± {tm_s:.2f}"
                else:
                    row_base = f"{name:<22} | {f1_m:<8.2f} | {prec_m:<8.2f} | {rec_m:<8.2f} | {mcc_m:<8.3f} | {fpr_m:<8.2f} | {tpr_m:<8.2f} | {tm_m:<10.2f}"
                    tsv_row = f"{name}\t{f1_m:.2f}\t{prec_m:.2f}\t{rec_m:.2f}\t{mcc_m:.3f}\t{fpr_m:.2f}\t{tpr_m:.2f}\t{tm_m:.2f}"
                    
                print(row_base)
                # Troca ponto por vírgula para planilhas em PT-BR
                tsv_lines.append(tsv_row.replace('.', ','))
            
            else: # Fluxo Retrocompatibilidade
                y_true_full = np.array(data.get('true_labels', data.get('y_true', [])))
                y_pred_full = np.array(data.get('predicted_classes', data.get('y_pred', [])))
                
                y_true_list = y_true_full[warmup_instances:] if len(y_true_full) > warmup_instances else y_true_full
                y_pred_list = y_pred_full[warmup_instances:] if len(y_pred_full) > warmup_instances else y_pred_full
                
                f1, prec, recall, mcc, fpr, tpr = self.calc_sklearn_metrics(y_true_list, y_pred_list, target_class)
                exec_time = data.get('exec_time', 0.0)

                print(f"{'-'*line_len}")
                row_base = f"{name:<22} | {f1:<8.2f} | {prec:<8.2f} | {recall:<8.2f} | {mcc:<8.3f} | {fpr:<8.2f} | {tpr:<8.2f} | {exec_time:<10.2f}"
                print(row_base)
                
                tsv_row = f"{name}\t{f1:.2f}\t{prec:.2f}\t{recall:.2f}\t{mcc:.3f}\t{fpr:.2f}\t{tpr:.2f}\t{exec_time:.2f}"
                tsv_lines.append(tsv_row.replace('.', ','))
        
        print(f"{'='*line_len}\n")
        
        # --- BLOCO ADICIONAL PARA CÓPIA RÁPIDA ---
        print("📋 CÓPIA PARA PLANILHA (Selecione as linhas abaixo e cole no Google Sheets/Excel):")
        for line in tsv_lines:
            print(line)
        print("\n")