import time
import numpy as np
from capymoa.evaluation import ClassificationEvaluator
from src.Anomaly.Results import Metrics, Plots

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

    def _run_anomaly_evaluation(self, stream, algorithms, window_size, title, warmup_instances=0, target_class=None, target_class_pass=None, threshold=0.5, ae_keywords=None, dataset_name="Cenario", recovery_window=1000):
       
        if ae_keywords is None:
            ae_keywords = ['AE', 'AUTOENCODER']

        results_metrics = {}
        results_scores = {}
        
        predictions_history = {}
        schema = stream.get_schema()
        
        min_warmup_required = max(warmup_instances, 0)

        for alg_idx, (alg_name, learner) in enumerate(algorithms.items()):
            start_time = time.time()
            stream.restart()
            evaluator_class = ClassificationEvaluator(schema=schema, window_size=window_size)
            
            history = {'instances': [], 'f1_score': [], 'precision': [], 'recall': []}
            results_scores[alg_name] = {'scores': []}
            
            alg_true_labels = []
            alg_true_labels_multi = []
            alg_predicted_classes = []
            
            count = 0
            is_normal_only_alg = any(kw.upper() in alg_name.upper() for kw in ae_keywords)

            while stream.has_more_instances():
                instance = stream.next_instance()
                
                true_label_multiclass = instance.y_index 
                true_label_binary = 1 if true_label_multiclass != self.normal_class_idx else 0
                
                is_warmup_phase = count < min_warmup_required

                score = learner.score_instance(instance) 
                results_scores[alg_name]['scores'].append(score)
                
                predicted_class = 1 if score > threshold else 0
                
                alg_true_labels.append(true_label_binary)
                alg_true_labels_multi.append(true_label_multiclass)
                alg_predicted_classes.append(predicted_class)
                
                if count >= min_warmup_required:
                    evaluator_class.update(true_label_binary, predicted_class)
               
                try:
                    if is_normal_only_alg:
                        if is_warmup_phase or predicted_class == 0:
                            learner.train(instance)
                    else:
                        learner.train(instance)
                except ValueError:
                    pass

                if count >= min_warmup_required and count > 0 and count % window_size == 0:
                    class_metrics = evaluator_class.metrics_dict()
                    f1_val = self.metrics.get_metric_classifier(class_metrics, 'f1_score', target_class=target_class)
                    prec_val = self.metrics.get_metric_classifier(class_metrics, 'precision', target_class=target_class)
                    recall_val = self.metrics.get_metric_classifier(class_metrics, 'recall', target_class=target_class)

                    history['instances'].append(count)
                    history['f1_score'].append(f1_val)
                    history['precision'].append(prec_val)
                    history['recall'].append(recall_val)
                        
                count += 1
                
            exec_time = time.time() - start_time
            results_metrics[alg_name] = history
            
            predictions_history[alg_name] = {
                'true_labels': alg_true_labels,
                'true_labels_multi': alg_true_labels_multi, 
                'predicted_classes': alg_predicted_classes,
                'exec_time': exec_time
            }

        primeiro_algoritmo = list(algorithms.keys())[0]
        y_true_array = np.array(predictions_history[primeiro_algoritmo]['true_labels'])
        y_true_multi = np.array(predictions_history[primeiro_algoritmo]['true_labels_multi'])
        
        attack_indices = np.where(y_true_array == 1)[0]
        
        attack_regions = []
        if len(attack_indices) > 0:
            start_idx = attack_indices[0]
            last_idx = attack_indices[0]
            for idx in attack_indices[1:]:
                if idx - last_idx > 1000:
                    block_labels = y_true_multi[start_idx:last_idx+1]
                    block_attack_labels = block_labels[block_labels != self.normal_class_idx]
                    
                    if len(block_attack_labels) > 0:
                        block_label = np.bincount(block_attack_labels).argmax()
                    else:
                        block_label = 1
                        
                    attack_regions.append((start_idx, last_idx, block_label))
                    start_idx = idx
                last_idx = idx
            
            block_labels = y_true_multi[start_idx:last_idx+1]
            block_attack_labels = block_labels[block_labels != self.normal_class_idx]
            if len(block_attack_labels) > 0:
                block_label = np.bincount(block_attack_labels).argmax()
            else:
                block_label = 1
            attack_regions.append((start_idx, last_idx, block_label))

        self.metrics.display_cumulative_metrics(
            predictions_history, 
            warmup_instances=min_warmup_required, 
            target_class=target_class,
            target_class_pass=target_class_pass,
            attack_regions=attack_regions,
            recovery_window=recovery_window,
            normal_class_idx=self.normal_class_idx
        )

        self.plots.plot_score(results_scores, attack_regions, title, threshold)
        self.plots.plot_metrics(results_metrics, attack_regions, title, window_size, target_class)

        dados_finais = predictions_history[primeiro_algoritmo]
        y_true_final = dados_finais['true_labels'][min_warmup_required:] if len(dados_finais['true_labels']) > min_warmup_required else dados_finais['true_labels']
        y_pred_final = dados_finais['predicted_classes'][min_warmup_required:] if len(dados_finais['predicted_classes']) > min_warmup_required else dados_finais['predicted_classes']
        
        f1_final, prec_final, recall_final, mcc_final, fpr_final, tpr_final = self.metrics.calc_sklearn_metrics(y_true_final, y_pred_final, target_class)
        
        return {
            'f1_score': f1_final,
            'precision': prec_final,
            'recall': recall_final,
            'mcc': mcc_final,
            'fpr': fpr_final,
            'tpr': tpr_final,
            'exec_time': dados_finais.get('exec_time', 0.0)
        }