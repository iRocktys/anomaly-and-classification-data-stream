import numpy as np
import time
from src.Results.Metrics import Metrics
from src.Results.Plots import Plots

class ClassificationExperimentRunner:
    def __init__(self, target_names=None, n_runs=1):
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.n_runs = n_runs
        self.normal_class_idx = 0
        for i, name in enumerate(self.target_names):
            if str(name).strip().upper() in ['BENIGN', 'NORMAL', '0']:
                self.normal_class_idx = i
                break
        self.metrics = Metrics()
        self.plots = Plots(self.target_names)

    def prequential_test(self, stream, learner, window_evaluation, warmup_instances):
        stream.restart()
        y_true_list, y_pred_list, true_labels_multi = [], [], []
        instances_list, f1_list, prec_list, rec_list = [], [], [], []
        fp_list, fn_list = [], []
        
        count = 0
        while stream.has_more_instances():
            instance = stream.next_instance()
            true_label_multiclass = instance.y_index 
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            prediction = learner.predict(instance)
            prediction = 0 if prediction is None else prediction
            binary_prediction = 1 if prediction > 0 else 0
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(binary_prediction)
            true_labels_multi.append(true_label_multiclass)
            
            learner.train(instance)
            
            if count >= warmup_instances and count > 0 and count % window_evaluation == 0:
                start_idx = max(warmup_instances, len(y_true_list) - window_evaluation)
                y_t_win = y_true_list[start_idx:]
                y_p_win = y_pred_list[start_idx:]
                
                f1_v, prec_v, rec_v, mcc_v, fp_v, fn_v = self.metrics.calc_sklearn_metrics(y_t_win, y_p_win)
                
                instances_list.append(count)
                f1_list.append(f1_v)
                prec_list.append(prec_v)
                rec_list.append(rec_v)
                fp_list.append(fp_v)
                fn_list.append(fn_v)
                    
            count += 1
            
        return {
            'y_true': y_true_list,
            'y_pred': y_pred_list,
            'true_labels_multi': true_labels_multi,
            'instances': instances_list,
            'f1': f1_list,
            'precision': prec_list,
            'recall': rec_list,
            'fp': fp_list,
            'fn': fn_list
        }

    def run_classification_evaluation(self, stream, algorithms, window_evaluation=1000, title="Avaliação Prequencial", warmup_instances=0, algorithm_params=None, is_optimized=True, num_features=None, exec_id="N/A"):
        predictions_history = {}
        
        param_type = "Otimizado" if is_optimized else "Default"
        feat_type = "FullFeatures" if (num_features is None or num_features > 50) else "33Features"
        final_scenario_name = f"{param_type}_{feat_type}"
        
        for alg_name, learner_or_factory in algorithms.items():
            runs_data = []
            exec_times = []
            
            print(f"\n[{alg_name}] Executando {self.n_runs} rodada(s) prequencial(is)...")
            for run in range(self.n_runs):
                start_time = time.time()
                current_seed = 42 + run
                
                if callable(learner_or_factory):
                    learner = learner_or_factory(run_seed=current_seed)
                else:
                    learner = learner_or_factory
                    if run > 0 and hasattr(learner, 'reset'):
                        learner.reset()
                    
                result = self.prequential_test(stream, learner, window_evaluation, warmup_instances)
                exec_times.append(time.time() - start_time)
                runs_data.append(result)
            
            f1_matrix = np.array([r['f1'] for r in runs_data])
            prec_matrix = np.array([r['precision'] for r in runs_data])
            rec_matrix = np.array([r['recall'] for r in runs_data])
            fp_matrix = np.array([r['fp'] for r in runs_data])
            fn_matrix = np.array([r['fn'] for r in runs_data])
            
            cum_metrics_list = []
            true_labels_multi = runs_data[0]['true_labels_multi']
            
            for r in runs_data:
                y_t = np.array(r['y_true'])[warmup_instances:]
                y_p = np.array(r['y_pred'])[warmup_instances:]
                cum_metrics_list.append(self.metrics.calc_sklearn_metrics(y_t, y_p))
                
            cum_matrix = np.array(cum_metrics_list) 
            
            predictions_history[alg_name] = {
                'instances': runs_data[0]['instances'],
                'f1_mean': np.mean(f1_matrix, axis=0), 'f1_std': np.std(f1_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(f1_matrix[0]),
                'precision_mean': np.mean(prec_matrix, axis=0), 'precision_std': np.std(prec_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(prec_matrix[0]),
                'recall_mean': np.mean(rec_matrix, axis=0), 'recall_std': np.std(rec_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(rec_matrix[0]),
                'fp_mean': np.mean(fp_matrix, axis=0), 'fp_std': np.std(fp_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(fp_matrix[0]),
                'fn_mean': np.mean(fn_matrix, axis=0), 'fn_std': np.std(fn_matrix, axis=0) if self.n_runs > 1 else np.zeros_like(fn_matrix[0]),
                'exec_time_mean': np.mean(exec_times), 'exec_time_std': np.std(exec_times) if self.n_runs > 1 else 0.0,
                
                'cumulative': {
                    'f1': (np.mean(cum_matrix[:, 0]), np.std(cum_matrix[:, 0]) if self.n_runs > 1 else 0.0),
                    'prec': (np.mean(cum_matrix[:, 1]), np.std(cum_matrix[:, 1]) if self.n_runs > 1 else 0.0),
                    'rec': (np.mean(cum_matrix[:, 2]), np.std(cum_matrix[:, 2]) if self.n_runs > 1 else 0.0),
                    'mcc': (np.mean(cum_matrix[:, 3]), np.std(cum_matrix[:, 3]) if self.n_runs > 1 else 0.0),
                    'fp': (np.mean(cum_matrix[:, 4]), np.std(cum_matrix[:, 4]) if self.n_runs > 1 else 0.0),
                    'fn': (np.mean(cum_matrix[:, 5]), np.std(cum_matrix[:, 5]) if self.n_runs > 1 else 0.0)
                },
                'true_labels_multi': true_labels_multi
            }

        first_algo = list(algorithms.keys())[0]
        attack_regions = self.metrics.extract_attack_regions(predictions_history[first_algo]['true_labels_multi'], normal_class_idx=self.normal_class_idx)
            
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history,
            warmup_instances=warmup_instances,
            n_runs=self.n_runs,
            params_dict=algorithm_params,
            experiment_name=title,
            scenario_name=final_scenario_name,
            discretization="N/A",
            window_evaluation=window_evaluation,
            exec_id=exec_id
        )
        
        self.plots.plot_metrics(
            results=predictions_history, 
            attack_regions=attack_regions, 
            title=title, 
            window_size=window_evaluation,
            scenario_name=final_scenario_name
        )
        
        self.plots.plot_fp_fn(
            results=predictions_history, 
            attack_regions=attack_regions, 
            title=title, 
            window_size=window_evaluation,
            scenario_name=final_scenario_name
        )