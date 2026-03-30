import numpy as np
import time
from capymoa.evaluation import ClassificationEvaluator
from src.Classification.Results import Metrics, Plots

class ClassificationExperimentRunner:
    def __init__(self, target_names=None):
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.metrics = Metrics()
        self.plots = Plots(self.target_names)

    def _get_metric_class(self, metrics_dict, metric_name, target_class=1):
        norm_metrics = {str(k).lower(): v for k, v in metrics_dict.items()}
        metric_name = str(metric_name).lower()
        
        if target_class is None or str(target_class).lower() == 'macro':
            val_0 = norm_metrics.get(f'{metric_name}_0', 0.0)
            val_1 = norm_metrics.get(f'{metric_name}_1', 0.0)
            
            val_0 = 0.0 if val_0 is None or np.isnan(val_0) else float(val_0)
            val_1 = 0.0 if val_1 is None or np.isnan(val_1) else float(val_1)
            
            return (val_0 + val_1) / 2.0
            
        else:
            val = norm_metrics.get(f'{metric_name}_{target_class}')
            return float(val) if val is not None and not np.isnan(val) else 0.0

    def run_classification_evaluation(self, stream, algorithms, window_size=1000, title="Avaliação Prequencial", warmup_instances=0, target_class=1, target_class_pass=None, recovery_window=1000):
        results = {}
        
        for name in algorithms:
            results[name] = {
                'instances': [],
                'f1': [], 'precision': [], 'recall': [], 
                'y_true': [], 'y_pred': [], 'true_labels_multi': [],
                'evaluator': ClassificationEvaluator(schema=stream.get_schema()),
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
                evaluator = res['evaluator']

                start_exec = time.time()
                prediction = model.predict(instance)
                if prediction is None:
                    prediction = 0
                
                if instance_idx >= warmup_instances:
                    evaluator.update(true_label_multiclass, prediction)
                
                binary_prediction = 1 if prediction > 0 else 0
                res['y_true'].append(binary_true_label)
                res['y_pred'].append(binary_prediction)
                res['true_labels_multi'].append(true_label_multiclass)

                model.train(instance)
                res['exec_time'] += (time.time() - start_exec)

                if instance_idx >= warmup_instances and instance_idx > 0 and instance_idx % window_size == 0:
                    res['instances'].append(instance_idx)
                    m_dict = evaluator.metrics_dict()
                    
                    res['precision'].append(self._get_metric_class(m_dict, 'precision', target_class))
                    res['recall'].append(self._get_metric_class(m_dict, 'recall', target_class))
                    res['f1'].append(self._get_metric_class(m_dict, 'f1_score', target_class))

            instance_idx += 1
        
        # Extrai as regiões de ataque de forma consolidada, respeitando a tolerância aos pacotes normais
        first_algo = list(algorithms.keys())[0]
        y_true_multi = results[first_algo]['true_labels_multi']
        attack_regions = self.metrics.extract_attack_regions(y_true_multi, normal_class_idx=0)
            
        self.metrics.display_cumulative_metrics(
            predictions_history=results,
            warmup_instances=warmup_instances,
            target_class=target_class,
            target_class_pass=target_class_pass,
            attack_regions=attack_regions,
            recovery_window=recovery_window
        )
        self.plots.plot_metrics(results, attack_regions=attack_regions, title=title, window_size=window_size)