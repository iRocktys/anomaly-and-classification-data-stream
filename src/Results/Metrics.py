import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
import warnings

warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")

class Metrics:
    def get_metric_classifier(self, metrics_dict, metric_name):
        norm_metrics = {str(k).lower(): v for k, v in metrics_dict.items()}
        metric_name = str(metric_name).lower()
        val = norm_metrics.get(f'{metric_name}_1')
        return float(val) if val is not None and not np.isnan(val) else 0.0

    def calc_sklearn_metrics(self, y_true, y_pred, target_class=None):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        f1 = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=1)
        prec = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=1)
        rec = recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=1)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        y_t_arr = np.array(y_true)
        y_p_arr = np.array(y_pred)
        fp = float(np.sum((y_p_arr == 1) & (y_t_arr == 0)))
        fn = float(np.sum((y_p_arr == 0) & (y_t_arr == 1)))
            
        return f1 * 100.0, prec * 100.0, rec * 100.0, mcc, fp, fn
    
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

    def display_cumulative_metrics(self, predictions_history, warmup_instances=0, target_class=None, n_runs=1, params_dict=None):
        titulo_relatorio = "RELATÓRIO DE MÉTRICAS | BINARY (CLASSE 1)"
        
        param_headers_print = []
        param_values_print = []
        
        # Tratamento dinâmico das colunas de parâmetros e seus valores
        if params_dict:
            for k, v in params_dict.items():
                k_str = str(k)
                if isinstance(v, float):
                    v_str = f"{v:.4f}"
                else:
                    v_str = str(v)
                
                # Alinhamento estético para o print do console
                width = max(len(k_str), len(v_str))
                param_headers_print.append(f"{k_str:<{width}}")
                param_values_print.append(f"{v_str:<{width}}")

        param_header_print_str = " | " + " | ".join(param_headers_print) if param_headers_print else ""
        param_val_print_str = " | " + " | ".join(param_values_print) if param_values_print else ""
        
        param_header_tsv_str = "\t" + "\t".join([str(k) for k in params_dict.keys()]) if params_dict else ""
        param_val_tsv_str = "\t" + "\t".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in params_dict.values()]) if params_dict else ""

        if n_runs > 1:
            header_base = f"{'Modelo/Algoritmo':<22} | {'F1 (%)':<17} | {'Prec (%)':<17} | {'Rec (%)':<17} | {'MCC':<17} | {'FP':<11} | {'FN':<11} | {'Tempo (s)':<15}{param_header_print_str}"
        else:
            header_base = f"{'Modelo/Algoritmo':<22} | {'F1 (%)':<10} | {'Prec (%)':<10} | {'Rec (%)':<10} | {'MCC':<10} | {'FP':<10} | {'FN':<10} | {'Tempo (s)':<12}{param_header_print_str}"
            
        line_len = max(len(header_base), len(titulo_relatorio) + 4)

        print(f"\n{'='*line_len}")
        print(f"{titulo_relatorio:^{line_len}}")
        print(f"{'='*line_len}")
        print(header_base)
        print(f"{'-'*line_len}")

        tsv_header = f"Modelo/Algoritmo\tF1 (%)\tPrec (%)\tRec (%)\tMCC\tFP\tFN\tTempo (s){param_header_tsv_str}"
        tsv_lines = [tsv_header]

        for name, data in predictions_history.items():
            if 'cumulative' in data:
                f1_m, f1_s = data['cumulative']['f1']
                prec_m, prec_s = data['cumulative']['prec']
                rec_m, rec_s = data['cumulative']['rec']
                mcc_m, mcc_s = data['cumulative']['mcc']
                fp_m, fp_s = data['cumulative']['fp']
                fn_m, fn_s = data['cumulative']['fn']
                tm_m, tm_s = data['exec_time_mean'], data['exec_time_std']
                
                # Arredonda sempre para cima (ceil) e transforma em Inteiro (int)
                fp_m_int = int(np.ceil(fp_m))
                fp_s_int = int(np.ceil(fp_s))
                fn_m_int = int(np.ceil(fn_m))
                fn_s_int = int(np.ceil(fn_s))
                
                print(f"{'-'*line_len}")
                if n_runs > 1:
                    row_base = f"{name:<22} | {f1_m:>7.4f} ± {f1_s:<6.4f} | {prec_m:>7.4f} ± {prec_s:<6.4f} | {rec_m:>7.4f} ± {rec_s:<6.4f} | {mcc_m:>7.4f} ± {mcc_s:<6.4f} | {fp_m_int:>4} ± {fp_s_int:<4} | {fn_m_int:>4} ± {fn_s_int:<4} | {tm_m:>6.4f} ± {tm_s:<5.4f}"
                    tsv_row = f"{name}\t{f1_m:.4f} ± {f1_s:.4f}\t{prec_m:.4f} ± {prec_s:.4f}\t{rec_m:.4f} ± {rec_s:.4f}\t{mcc_m:.4f} ± {mcc_s:.4f}\t{fp_m_int} ± {fp_s_int}\t{fn_m_int} ± {fn_s_int}\t{tm_m:.4f} ± {tm_s:.4f}"
                else:
                    row_base = f"{name:<22} | {f1_m:<10.4f} | {prec_m:<10.4f} | {rec_m:<10.4f} | {mcc_m:<10.4f} | {fp_m_int:<10} | {fn_m_int:<10} | {tm_m:<12.4f}"
                    tsv_row = f"{name}\t{f1_m:.4f}\t{prec_m:.4f}\t{rec_m:.4f}\t{mcc_m:.4f}\t{fp_m_int}\t{fn_m_int}\t{tm_m:.4f}"
                
                row_base += param_val_print_str
                tsv_row += param_val_tsv_str
                
                print(row_base)
                tsv_lines.append(tsv_row.replace('.', ','))
            
            else: 
                y_true_full = np.array(data.get('true_labels', data.get('y_true', [])))
                y_pred_full = np.array(data.get('predicted_classes', data.get('y_pred', [])))
                
                y_true_list = y_true_full[warmup_instances:] if len(y_true_full) > warmup_instances else y_true_full
                y_pred_list = y_pred_full[warmup_instances:] if len(y_pred_full) > warmup_instances else y_pred_full
                
                f1, prec, recall, mcc, fp, fn = self.calc_sklearn_metrics(y_true_list, y_pred_list)
                exec_time = data.get('exec_time', 0.0)
                
                fp_int = int(np.ceil(fp))
                fn_int = int(np.ceil(fn))

                print(f"{'-'*line_len}")
                row_base = f"{name:<22} | {f1:<10.4f} | {prec:<10.4f} | {recall:<10.4f} | {mcc:<10.4f} | {fp_int:<10} | {fn_int:<10} | {exec_time:<12.4f}"
                tsv_row = f"{name}\t{f1:.4f}\t{prec:.4f}\t{recall:.4f}\t{mcc:.4f}\t{fp_int}\t{fn_int}\t{exec_time:.4f}"
                
                row_base += param_val_print_str
                tsv_row += param_val_tsv_str
                
                print(row_base)
                tsv_lines.append(tsv_row.replace('.', ','))
        
        print(f"{'='*line_len}\n")
        
        print("📋 CÓPIA PARA PLANILHA (Selecione as linhas abaixo e cole no Google Sheets/Excel):")
        for line in tsv_lines:
            print(line)
        print("\n")