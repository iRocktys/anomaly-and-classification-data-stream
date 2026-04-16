import numpy as np
import os
import csv
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
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

    def display_cumulative_metrics(self, predictions_history, warmup_instances=0, target_class=None, n_runs=1, params_dict=None, experiment_name="General", scenario_name="Default_Full", discretization=None, window_evaluation=None, exec_id="N/A"):
        titulo_relatorio = f"METRICS REPORT | {scenario_name.upper()}"
        
        discretization_str = f"{discretization:.4f}" if isinstance(discretization, (float, int)) else str(discretization) if discretization is not None else "-"
        window_eval_str = str(window_evaluation) if window_evaluation is not None else "-"
        exec_id_str = str(exec_id)

        parts = str(experiment_name).split('_')
        if len(parts) >= 2:
            category = parts[0]
            contamination_block = parts[1]
        else:
            category = experiment_name
            contamination_block = "-"

        param_headers_print = []
        param_values_print = []
        clean_params = {}
        
        if params_dict:
            clean_params = {k: v for k, v in params_dict.items() if isinstance(v, (int, float, str, bool))}
            for k, v in clean_params.items():
                k_str, v_str = str(k), (f"{v:.4f}" if isinstance(v, float) else str(v))
                width = max(len(k_str), len(v_str))
                param_headers_print.append(f"{k_str:<{width}}")
                param_values_print.append(f"{v_str:<{width}}")

        param_header_print_str = " | " + " | ".join(param_headers_print) if param_headers_print else ""
        param_val_print_str = " | " + " | ".join(param_values_print) if param_values_print else ""

        header_base = f"{'Algorithm':<22} | {'F1 (%)':<17} | {'Prec (%)':<17} | {'Rec (%)':<17} | {'MCC':<17} | {'FP':<11} | {'FN':<11} | {'Time (s)':<15} | {'Discrtz':<10} | {'Win_Eval':<10}{param_header_print_str}"
        line_len = max(len(header_base), len(titulo_relatorio) + 4)

        print(f"\n{'='*line_len}\n{titulo_relatorio:^{line_len}}\n{'='*line_len}\n{header_base}\n{'-'*line_len}")

        for name, data in predictions_history.items():
            algo_dir = os.path.join("output", "Metrics", name)
            os.makedirs(algo_dir, exist_ok=True)
            csv_file_path = os.path.join(algo_dir, f"{name}_{scenario_name}.csv")
            
            file_exists = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0

            if 'cumulative' in data:
                f1_m, f1_s = data['cumulative']['f1']
                prec_m, prec_s = data['cumulative']['prec']
                rec_m, rec_s = data['cumulative']['rec']
                mcc_m, mcc_s = data['cumulative']['mcc']
                fp_m, fp_s = data['cumulative']['fp']
                fn_m, fn_s = data['cumulative']['fn']
                tm_m, tm_s = data['exec_time_mean'], data['exec_time_std']
                
                headers = ['Exec_ID', 'Dataset', 'Category', 'Contamination_Block', 'F1_avg', 'F1_std', 'Prec_avg', 'Prec_std', 'Rec_avg', 'Rec_std', 'MCC_avg', 'MCC_std', 'FP_avg', 'FP_std', 'FN_avg', 'FN_std', 'Time_avg', 'Time_std', 'Discretization', 'Win_Evaluation']
                metrics_row = [exec_id_str, experiment_name, category, contamination_block, f"{f1_m:.4f}", f"{f1_s:.4f}", f"{prec_m:.4f}", f"{prec_s:.4f}", 
                               f"{rec_m:.4f}", f"{rec_s:.4f}", f"{mcc_m:.4f}", f"{mcc_s:.4f}", 
                               int(np.ceil(fp_m)), int(np.ceil(fp_s)), int(np.ceil(fn_m)), int(np.ceil(fn_s)), 
                               f"{tm_m:.4f}", f"{tm_s:.4f}", discretization_str, window_eval_str]
                
                print(f"{name:<22} | {f1_m:>7.4f} ± {f1_s:<6.4f} | {prec_m:>7.4f} ± {prec_s:<6.4f} | {rec_m:>7.4f} ± {rec_s:<6.4f} | {mcc_m:>7.4f} ± {mcc_s:<6.4f} | {int(np.ceil(fp_m)):>4} | {int(np.ceil(fn_m)):>4} | {tm_m:>6.4f} | {discretization_str:<10} | {window_eval_str:<10}{param_val_print_str}")
            else:
                f1, pr, re, mcc, fp, fn = self.calc_sklearn_metrics(data.get('y_true', []), data.get('y_pred', []))
                tm = data.get('exec_time', 0.0)
                
                headers = ['Exec_ID', 'Dataset', 'Category', 'Contamination_Block', 'F1', 'F1_std', 'Prec', 'Prec_std', 'Rec', 'Rec_std', 'MCC', 'MCC_std', 'FP', 'FP_std', 'FN', 'FN_std', 'Time', 'Time_std', 'Discretization', 'Win_Evaluation']
                metrics_row = [exec_id_str, experiment_name, category, contamination_block, f"{f1:.4f}", "0", f"{pr:.4f}", "0", f"{re:.4f}", "0", f"{mcc:.4f}", "0", int(np.ceil(fp)), "0", int(np.ceil(fn)), "0", f"{tm:.4f}", "0", discretization_str, window_eval_str]
                
                print(f"{name:<22} | {f1:<10.4f} | {pr:<10.4f} | {re:<10.4f} | {mcc:<10.4f} | {int(np.ceil(fp)):<10} | {int(np.ceil(fn)):<10} | {tm:<12.4f} | {discretization_str:<10} | {window_eval_str:<10}{param_val_print_str}")

            if clean_params:
                for k, v in clean_params.items():
                    if not file_exists: headers.append(f"param_{k}")
                    metrics_row.append(f"{v:.4f}" if isinstance(v, float) else str(v))

            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                if not file_exists: writer.writerow(headers)
                writer.writerow([str(x).replace('.', ',') for x in metrics_row])

        print(f"{'-'*line_len}\nFile: output/Metrics/{name}/{name}_{scenario_name}.csv\n")