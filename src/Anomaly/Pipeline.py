import time
import numpy as np
from src.Results.Metrics import Metrics
from src.Results.Plots import Plots

class AnomalyExperimentRunner:
    def __init__(self, target_names):
        self.target_names = target_names
        self.normal_class_idx = 0
        for i, name in enumerate(target_names):
            if str(name).strip().upper() in ['BENIGN', 'NORMAL', '0']:
                self.normal_class_idx = i
                break
                
        self.metrics = Metrics()
        self.plots = Plots(target_names)

    def run_anomaly_evaluation(self, stream, algorithms, window_size, title, warmup_instances=0, target_class=None, target_class_pass=None, threshold=0.5, ae_keywords=None, dataset_name="Cenario", recovery_window=1000):
        if ae_keywords is None:
            ae_keywords = ['AE', 'AUTOENCODER']

        predictions_history = {}
        
        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            start_time = time.time()
            stream.restart()
            
            is_ae = any(kw.upper() in alg_name.upper() for kw in ae_keywords)
            
            y_true_list = []
            y_pred_list = []
            true_labels_multi = []
            scores = []
            
            # Listas para guardar os dados do gráfico de métricas evolutivas
            instances_list = []
            f1_list = []
            prec_list = []
            rec_list = []
            
            count = 0
            while stream.has_more_instances():
                instance = stream.next_instance()
                true_label_multiclass = instance.y_index
                binary_true_label = 1 if true_label_multiclass > 0 else 0
                
                score = learner.score_instance(instance) 
                
                if threshold == 'params':
                    pred = learner.predict(instance)
                    predicted_class = 1 if (pred is not None and pred > 0) else 0
                else:
                    predicted_class = 1 if score > threshold else 0
                
                scores.append(score)
                y_true_list.append(binary_true_label)
                y_pred_list.append(predicted_class)
                true_labels_multi.append(true_label_multiclass)
                
                try:
                    if not is_ae or predicted_class == 0:
                        learner.train(instance)
                except ValueError:
                    pass
                
                # Coleta as métricas evolutivas a cada "window_size" para desenhar o gráfico
                if count >= warmup_instances and count > 0 and count % window_size == 0:
                    y_t_win = y_true_list[warmup_instances:]
                    y_p_win = y_pred_list[warmup_instances:]
                    f1_v, prec_v, rec_v, *_ = self.metrics.calc_sklearn_metrics(y_t_win, y_p_win, target_class)
                    
                    instances_list.append(count)
                    f1_list.append(f1_v)
                    prec_list.append(prec_v)
                    rec_list.append(rec_v)
                        
                count += 1
                
            exec_time = time.time() - start_time
            
            predictions_history[alg_name] = {
                'y_true': y_true_list,
                'y_pred': y_pred_list,
                'true_labels_multi': true_labels_multi,
                'scores': scores,
                'instances': instances_list,
                'f1': f1_list,
                'precision': prec_list,
                'recall': rec_list,
                'exec_time': exec_time
            }

        first_algo = list(algorithms.keys())[0]
        y_true_multi = predictions_history[first_algo]['true_labels_multi']
        attack_regions = self.metrics.extract_attack_regions(y_true_multi, normal_class_idx=self.normal_class_idx)
        
        # Tabela de Métricas Acumulativas
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history,
            warmup_instances=warmup_instances,
            target_class=target_class,
            target_class_pass=target_class_pass,
            attack_regions=attack_regions,
            recovery_window=recovery_window,
            normal_class_idx=self.normal_class_idx
        )
        
        # Gráfico de Métricas Evolutivas (F1, Precision, Recall)
        self.plots.plot_metrics(
            results=predictions_history,
            attack_regions=attack_regions,
            title=f"Métricas Evolutivas - {title}",
            window_size=window_size,
            target_class=target_class
        )
        
        # Gráfico de Score de Anomalia
        self.plots.plot_score(
            results=predictions_history, 
            attack_regions=attack_regions, 
            title=f"Scores - {title}", 
            threshold=threshold if threshold != 'params' else 0.5
        )