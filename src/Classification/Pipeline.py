import numpy as np
import time
# IMPORTANTE: Garanta que este importe está apontando para o arquivo NOVO e UNIFICADO
from src.Results.Metrics import Metrics
from src.Results.Plots import Plots

class ClassificationExperimentRunner:
    def __init__(self, target_names=None):
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.metrics = Metrics()
        self.plots = Plots(self.target_names)

    def run_classification_evaluation(self, stream, algorithms, window_size=1000, title="Avaliação Prequencial", warmup_instances=0, target_class=1, target_class_pass=None, recovery_window=1000):
        results = {}
        
        for name in algorithms:
            results[name] = {
                'instances': [],
                'f1': [], 'precision': [], 'recall': [], 
                'mcc': [], 'fpr': [], 'tpr': [],
                'y_true': [], 'y_pred': [], 'true_labels_multi': [],
                'exec_time': 0.0
            }

        stream.restart()
        instance_idx = 0 

        while stream.has_more_instances():
            instance = stream.next_instance()
            true_label_multiclass = instance.y_index 
            binary_true_label = 1 if true_label_multiclass > 0 else 0

            for name, model in algorithms.items():
                res = results[name]

                start_exec = time.time()
                prediction = model.predict(instance)
                if prediction is None:
                    prediction = 0
                
                binary_prediction = 1 if prediction > 0 else 0
                res['y_true'].append(binary_true_label)
                res['y_pred'].append(binary_prediction)
                res['true_labels_multi'].append(true_label_multiclass)

                model.train(instance)
                res['exec_time'] += (time.time() - start_exec)

                if instance_idx >= warmup_instances and instance_idx > 0 and instance_idx % window_size == 0:
                    res['instances'].append(instance_idx)
                    
                    y_t_win = res['y_true'][warmup_instances:]
                    y_p_win = res['y_pred'][warmup_instances:]
                    
                    # Usa a função do Metrics.py unificado que retorna os 6 valores
                    f1_v, prec_v, rec_v, mcc_v, fpr_v, tpr_v = self.metrics.calc_sklearn_metrics(y_t_win, y_p_win, target_class)
                    
                    res['f1'].append(f1_v)
                    res['precision'].append(prec_v)
                    res['recall'].append(rec_v)
                    res['mcc'].append(mcc_v)
                    res['fpr'].append(fpr_v)
                    res['tpr'].append(tpr_v)

            instance_idx += 1
        
        first_algo = list(algorithms.keys())[0]
        y_true_multi = results[first_algo]['true_labels_multi']
        
        # Pega o índice da classe normal dinamicamente a partir dos nomes alvo
        normal_idx = 0
        for i, name in enumerate(self.target_names):
            if str(name).strip().upper() in ['BENIGN', 'NORMAL', '0']:
                normal_idx = i
                break

        attack_regions = self.metrics.extract_attack_regions(y_true_multi, normal_class_idx=normal_idx)
            
        self.metrics.display_cumulative_metrics(
            predictions_history=results,
            warmup_instances=warmup_instances,
            target_class=target_class,
            target_class_pass=target_class_pass,
            attack_regions=attack_regions,
            recovery_window=recovery_window,
            normal_class_idx=normal_idx
        )
        
        # Usa a função plot_metrics unificada, que plota na ordem: F1, Precision, Recall
        self.plots.plot_metrics(
            results=results, 
            attack_regions=attack_regions, 
            title=title, 
            window_size=window_size, 
            target_class=target_class
        )