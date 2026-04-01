import time
import numpy as np
from src.Results.Metrics import Metrics
from src.Results.Plots import Plots

class AnomalyExperimentRunner:
    def __init__(self, target_names, n_runs=1):
        self.target_names = target_names
        self.n_runs = n_runs
        self.normal_class_idx = 0
        for i, name in enumerate(target_names):
            if str(name).strip().upper() in ['BENIGN', 'NORMAL', '0']:
                self.normal_class_idx = i
                break
                
        self.metrics = Metrics()
        self.plots = Plots(target_names)

    def prequential_test(self, stream, learner, threshold, is_ae, window_size, warmup_instances, target_class):
        stream.restart()
        y_true_list, y_pred_list, true_labels_multi, scores = [], [], [], []
        instances_list, f1_list, prec_list, rec_list = [], [], [], []
        
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
            
            if count >= warmup_instances and count > 0 and count % window_size == 0:
                y_t_win = y_true_list[warmup_instances:]
                y_p_win = y_pred_list[warmup_instances:]
                f1_v, prec_v, rec_v, *_ = self.metrics.calc_sklearn_metrics(y_t_win, y_p_win, target_class)
                
                instances_list.append(count)
                f1_list.append(f1_v)
                prec_list.append(prec_v)
                rec_list.append(rec_v)
                    
            count += 1
            
        return {
            'y_true': y_true_list,
            'y_pred': y_pred_list,
            'true_labels_multi': true_labels_multi,
            'scores': scores,
            'instances': instances_list,
            'f1': f1_list,
            'precision': prec_list,
            'recall': rec_list
        }

    def _aggregate_behavioral(self, beh_metrics_list):
        if not beh_metrics_list or not beh_metrics_list[0]:
            return []
            
        n_attacks = len(beh_metrics_list[0])
        aggregated = []
        
        for i in range(n_attacks):
            passagens = [run[i]['passagem'] for run in beh_metrics_list]
            recuperacoes = [run[i]['recuperacao'] for run in beh_metrics_list]
            
            aggregated.append({
                'ataque_idx': beh_metrics_list[0][i]['ataque_idx'],
                'passagem': (np.mean(passagens), np.std(passagens) if self.n_runs > 1 else 0.0),
                'recuperacao': (np.mean(recuperacoes), np.std(recuperacoes) if self.n_runs > 1 else 0.0)
            })
        return aggregated

    def run_anomaly_evaluation(self, stream, algorithms, window_size, title, warmup_instances=0, target_class=None, target_class_pass=None, threshold=0.5, ae_keywords=None, dataset_name="Cenario", recovery_window=1000):
        if ae_keywords is None:
            ae_keywords = ['AE', 'AUTOENCODER']

        predictions_history = {}
        
        for alg_name, learner in algorithms.items():
            is_ae = any(kw.upper() in alg_name.upper() for kw in ae_keywords)
            runs_data = []
            exec_times = []
            
            print(f"\n[{alg_name}] Executando {self.n_runs} rodada(s) prequencial(is)...")
            for run in range(self.n_runs):
                start_time = time.time()
                
                if run > 0 and hasattr(learner, 'reset'):
                    learner.reset()
                    
                result = self.prequential_test(stream, learner, threshold, is_ae, window_size, warmup_instances, target_class)
                exec_times.append(time.time() - start_time)
                runs_data.append(result)
            
            f1_matrix = np.array([r['f1'] for r in runs_data])
            prec_matrix = np.array([r['precision'] for r in runs_data])
            rec_matrix = np.array([r['recall'] for r in runs_data])
            scores_matrix = np.array([r['scores'] for r in runs_data])
            
            cum_metrics_list = []
            beh_metrics_list = []
            
            true_labels_multi = runs_data[0]['true_labels_multi']
            attack_regions = self.metrics.extract_attack_regions(true_labels_multi, normal_class_idx=self.normal_class_idx)
            
            for r in runs_data:
                y_t = np.array(r['y_true'])[warmup_instances:]
                y_p = np.array(r['y_pred'])[warmup_instances:]
                cum_metrics_list.append(self.metrics.calc_sklearn_metrics(y_t, y_p, target_class))
                
                beh = self.metrics.calc_behavioral_metrics(r['y_true'], r['y_pred'], attack_regions, recovery_window, warmup_instances, target_class_pass)
                beh_metrics_list.append(beh)
                
            cum_matrix = np.array(cum_metrics_list) 
            
            predictions_history[alg_name] = {
                'instances': runs_data[0]['instances'],
                'f1_mean': np.mean(f1_matrix, axis=0), 'f1_std': np.std(f1_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(f1_matrix[0]),
                'precision_mean': np.mean(prec_matrix, axis=0), 'precision_std': np.std(prec_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(prec_matrix[0]),
                'recall_mean': np.mean(rec_matrix, axis=0), 'recall_std': np.std(rec_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(rec_matrix[0]),
                'scores_mean': np.mean(scores_matrix, axis=0), 'scores_std': np.std(scores_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(scores_matrix[0]),
                'exec_time_mean': np.mean(exec_times), 'exec_time_std': np.std(exec_times) if self.n_runs > 1 else 0.0,
                
                'cumulative': {
                    'f1': (np.mean(cum_matrix[:, 0]), np.std(cum_matrix[:, 0]) if self.n_runs > 1 else 0.0),
                    'prec': (np.mean(cum_matrix[:, 1]), np.std(cum_matrix[:, 1]) if self.n_runs > 1 else 0.0),
                    'rec': (np.mean(cum_matrix[:, 2]), np.std(cum_matrix[:, 2]) if self.n_runs > 1 else 0.0),
                    'mcc': (np.mean(cum_matrix[:, 3]), np.std(cum_matrix[:, 3]) if self.n_runs > 1 else 0.0),
                    'fpr': (np.mean(cum_matrix[:, 4]), np.std(cum_matrix[:, 4]) if self.n_runs > 1 else 0.0),
                    'tpr': (np.mean(cum_matrix[:, 5]), np.std(cum_matrix[:, 5]) if self.n_runs > 1 else 0.0)
                },
                'behavioral': self._aggregate_behavioral(beh_metrics_list),
                'true_labels_multi': true_labels_multi
            }

        first_algo = list(algorithms.keys())[0]
        attack_regions = self.metrics.extract_attack_regions(predictions_history[first_algo]['true_labels_multi'], normal_class_idx=self.normal_class_idx)
        
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history, target_class=target_class, target_class_pass=target_class_pass,
            attack_regions=attack_regions, recovery_window=recovery_window, normal_class_idx=self.normal_class_idx
        )
        self.plots.plot_metrics(results=predictions_history, attack_regions=attack_regions, title=f"Métricas Evolutivas - {title}", window_size=window_size, target_class=target_class)
        self.plots.plot_score(results=predictions_history, attack_regions=attack_regions, title=f"Scores - {title}", threshold=threshold if threshold != 'params' else 0.5)