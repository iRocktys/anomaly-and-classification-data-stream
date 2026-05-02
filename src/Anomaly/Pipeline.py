import time
import os
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

    def _parse_decision_strategy(self, decision_strategy):
        strategy = str(decision_strategy or "raw").strip().lower()

        if strategy == "raw":
            return {"name": "raw", "type": "raw", "window": None, "k": None, "n": None}
        if strategy.startswith("moving_average_w"):
            window = int(strategy.split("_w")[-1])
            return {"name": strategy, "type": "moving_average", "window": window, "k": None, "n": None}
        if strategy.startswith("persistence_") and "_of_" in strategy:
            left, right = strategy.replace("persistence_", "").split("_of_")
            k, n = int(left), int(right)
            if k <= 0 or n <= 0 or k > n:
                raise ValueError("A persistência precisa obedecer 0 < k <= n.")
            return {"name": strategy, "type": "persistence", "window": None, "k": k, "n": n}
        raise ValueError(
            "decision_strategy deve ser uma destas: raw, moving_average_w3, "
            "moving_average_w5, persistence_2_of_3, persistence_3_of_5."
        )

    def _combined_strategy_name(self, threshold_strategy, decision_config):
        return os.path.join(str(threshold_strategy), decision_config["name"])

    def _causal_moving_average(self, values, window):
        if not values:
            return 0.0
        recent = values[-max(1, int(window)):]
        return float(np.mean(recent))

    def _build_causal_moving_average_series(self, values, window):
        values = list(values)
        if not values:
            return []
        return [float(np.mean(values[max(0, i - window + 1):i + 1])) for i in range(len(values))]

    def _compute_z_threshold(self, warmup_scores, z_value, decision_config):
        if decision_config["type"] == "moving_average":
            reference_scores = self._build_causal_moving_average_series(warmup_scores, decision_config["window"])
        else:
            reference_scores = list(warmup_scores)
        if not reference_scores:
            return 0.5, None, None
        mu = float(np.mean(reference_scores))
        std = float(np.std(reference_scores))
        return mu + (float(z_value) * std), mu, std

    def _apply_decision_rule(self, score, threshold, score_history, peak_history, decision_config):
        if decision_config["type"] == "moving_average":
            decision_score = self._causal_moving_average(score_history, decision_config["window"])
            return 1 if decision_score > threshold else 0
        if decision_config["type"] == "persistence":
            is_peak = 1 if score > threshold else 0
            peak_history.append(is_peak)
            recent = peak_history[-decision_config["n"]:]
            return 1 if sum(recent) >= decision_config["k"] else 0
        return 1 if score > threshold else 0

    def prequential_test(self, stream, learner, discretization, is_ae, window_evaluation, warmup_instances, z_value=None, decision_strategy="raw"):
        decision_config = self._parse_decision_strategy(decision_strategy)
        stream.restart()
        y_true_list, y_pred_list, true_labels_multi, scores = [], [], [], []
        instances_list, f1_list, prec_list, rec_list = [], [], [], []
        fp_list, fn_list = [], []
        warmup_scores = []
        score_history = []
        peak_history = []
        
        run_threshold = discretization if isinstance(discretization, (float, int)) else 0.5
        count = 0
        calc_mu, calc_std = None, None
        
        while stream.has_more_instances():
            instance = stream.next_instance()
            true_label_multiclass = instance.y_index
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            score = learner.score_instance(instance) 
            scores.append(score)
            score_history.append(score)
            true_labels_multi.append(true_label_multiclass)
            
            if count < warmup_instances:
                warmup_scores.append(score)
                predicted_class = 0
                try:
                    learner.train(instance)
                except ValueError:
                    pass
            else:
                if count == warmup_instances and z_value is not None and warmup_instances > 0:
                    run_threshold, calc_mu, calc_std = self._compute_z_threshold(warmup_scores, z_value, decision_config)

                if discretization == 'params':
                    pred = learner.predict(instance)
                    predicted_class = 1 if (pred is not None and pred > 0) else 0
                else:
                    predicted_class = self._apply_decision_rule(score, run_threshold, score_history, peak_history, decision_config)
                
                try:
                    if not is_ae or predicted_class == 0:
                        learner.train(instance)
                except ValueError:
                    pass
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(predicted_class)
            
            if count >= warmup_instances and count > 0 and count % window_evaluation == 0:
                start_idx = max(warmup_instances, len(y_true_list) - window_evaluation)
                y_t_win = y_true_list[start_idx:]
                y_p_win = y_pred_list[start_idx:]

                # Métricas janeladas: mostram a variação local de desempenho em cada bloco,
                # permitindo observar quedas/subidas nas regiões de ataque.
                f1_v, prec_v, rec_v, _, fp, fn = self.metrics.calc_sklearn_metrics(y_t_win, y_p_win)
                
                instances_list.append(count)
                f1_list.append(f1_v)
                prec_list.append(prec_v)
                rec_list.append(rec_v)
                fp_list.append(fp)
                fn_list.append(fn)
                    
            count += 1
            
        return {
            'y_true': y_true_list,
            'y_pred': y_pred_list,
            'true_labels_multi': true_labels_multi,
            'scores': scores,
            'instances': instances_list,
            'f1': f1_list,
            'precision': prec_list,
            'recall': rec_list,
            'fp': fp_list,
            'fn': fn_list,
            'z_stats': {'mu': calc_mu, 'std': calc_std, 'threshold': run_threshold}
        }

    def run_anomaly_evaluation(self, stream, algorithms, window_evaluation=1000, title="Avaliação Prequencial", warmup_instances=0, discretization=0.5, ae_keywords=None, algorithm_params=None, is_optimized=True, num_features=None, exec_id="N/A", decision_strategy="raw"):
        if ae_keywords is None:
            ae_keywords = ['AE', 'AUTOENCODER']

        decision_config = self._parse_decision_strategy(decision_strategy)
        predictions_history = {}
        
        param_type = "Optimized" if is_optimized else "Default"
        feat_type = "FullFeatures" if (num_features is None or num_features > 50) else "33Features"
        final_scenario_name = f"{param_type}_{feat_type}"
        
        if not is_optimized:
            z_value = None
            strategy_name = 'fixed'
        else:
            z_value = algorithm_params.get('z') if algorithm_params else None
            
            if discretization == 'params':
                strategy_name = 'params'
            elif discretization == 'dinamic':
                strategy_name = 'dinamic'
            elif algorithm_params and 'z' in algorithm_params:
                strategy_name = 'z_score'
            else:
                strategy_name = 'fixed'
        
        for alg_name, learner_or_factory in algorithms.items():
            is_ae = any(kw.upper() in alg_name.upper() for kw in ae_keywords)
            runs_data = []
            exec_times = []
            mu_list, std_list, thresh_list = [], [], []
            
            print(f"\n[{alg_name}] Executando {self.n_runs} rodada(s) prequencial(is) | Decisão: {decision_config['name']}...")
            for run in range(self.n_runs):
                start_time = time.time()
                current_seed = 42 + run
                
                if callable(learner_or_factory):
                    learner = learner_or_factory(run_seed=current_seed)
                else:
                    learner = learner_or_factory
                    if run > 0 and hasattr(learner, 'reset'):
                        learner.reset()
                    
                result = self.prequential_test(stream, learner, discretization, is_ae, window_evaluation, warmup_instances, z_value=z_value, decision_strategy=decision_config["name"])
                exec_times.append(time.time() - start_time)
                runs_data.append(result)
                
                if result['z_stats']['threshold'] is not None:
                    mu_list.append(result['z_stats']['mu'])
                    std_list.append(result['z_stats']['std'])
                    thresh_list.append(result['z_stats']['threshold'])
            
            f1_matrix = np.array([r['f1'] for r in runs_data])
            prec_matrix = np.array([r['precision'] for r in runs_data])
            rec_matrix = np.array([r['recall'] for r in runs_data])
            fp_matrix = np.array([r['fp'] for r in runs_data])
            fn_matrix = np.array([r['fn'] for r in runs_data])
            scores_matrix = np.array([r['scores'] for r in runs_data])
            
            cum_metrics_list = []
            true_labels_multi = runs_data[0]['true_labels_multi']
            
            for r in runs_data:
                y_t = np.array(r['y_true'])[warmup_instances:] if len(r['y_true']) > warmup_instances else np.array(r['y_true'])
                y_p = np.array(r['y_pred'])[warmup_instances:] if len(r['y_pred']) > warmup_instances else np.array(r['y_pred'])
                cum_metrics_list.append(self.metrics.calc_sklearn_metrics(y_t, y_p))
                
            cum_matrix = np.array(cum_metrics_list) 
            
            predictions_history[alg_name] = {
                'instances': runs_data[0]['instances'],
                'f1_mean': np.mean(f1_matrix, axis=0) if len(f1_matrix[0]) > 0 else [],
                'f1_std': np.std(f1_matrix, axis=0) if self.n_runs > 1 and len(f1_matrix[0]) > 0 else (np.zeros_like(f1_matrix[0]) if len(f1_matrix[0]) > 0 else []),
                'precision_mean': np.mean(prec_matrix, axis=0) if len(prec_matrix[0]) > 0 else [],
                'precision_std': np.std(prec_matrix, axis=0) if self.n_runs > 1 and len(prec_matrix[0]) > 0 else (np.zeros_like(prec_matrix[0]) if len(prec_matrix[0]) > 0 else []),
                'recall_mean': np.mean(rec_matrix, axis=0) if len(rec_matrix[0]) > 0 else [],
                'recall_std': np.std(rec_matrix, axis=0) if self.n_runs > 1 and len(rec_matrix[0]) > 0 else (np.zeros_like(rec_matrix[0]) if len(rec_matrix[0]) > 0 else []),
                'fp_mean': np.mean(fp_matrix, axis=0) if len(fp_matrix[0]) > 0 else [],
                'fp_std': np.std(fp_matrix, axis=0) if self.n_runs > 1 and len(fp_matrix[0]) > 0 else (np.zeros_like(fp_matrix[0]) if len(fp_matrix[0]) > 0 else []),
                'fn_mean': np.mean(fn_matrix, axis=0) if len(fn_matrix[0]) > 0 else [],
                'fn_std': np.std(fn_matrix, axis=0) if self.n_runs > 1 and len(fn_matrix[0]) > 0 else (np.zeros_like(fn_matrix[0]) if len(fn_matrix[0]) > 0 else []),
                'scores_mean': np.mean(scores_matrix, axis=0) if len(scores_matrix[0]) > 0 else [],
                'scores_std': np.std(scores_matrix, axis=0) if self.n_runs > 1 and len(scores_matrix[0]) > 0 else (np.zeros_like(scores_matrix[0]) if len(scores_matrix[0]) > 0 else []),
                'exec_time_mean': np.mean(exec_times),
                'exec_time_std': np.std(exec_times) if self.n_runs > 1 else 0.0,
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
        
        display_params = dict(algorithm_params) if algorithm_params else {}
        display_params['decision_strategy'] = decision_config['name']
        display_params['decision_type'] = decision_config['type']
        display_params['decision_window'] = decision_config.get('window')
        display_params['persistence_k'] = decision_config.get('k')
        display_params['persistence_n'] = decision_config.get('n')
        if z_value is not None and thresh_list:
            display_params['u'] = float(np.mean(mu_list))
            display_params['std'] = float(np.mean(std_list))
            final_discretization = float(np.mean(thresh_list))
        else:
            final_discretization = discretization

        if not is_optimized:
            output_strategy_name = 'fixed'
        else:
            output_strategy_name = self._combined_strategy_name(strategy_name, decision_config)
        
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history,
            warmup_instances=warmup_instances,
            n_runs=self.n_runs,
            params_dict=display_params,
            experiment_name=title,
            scenario_name=final_scenario_name,
            discretization=final_discretization,
            window_evaluation=window_evaluation,
            exec_id=exec_id,
            discretization_strategy=output_strategy_name,
            task_type="anomaly",
            threshold_strategy=strategy_name,
            decision_strategy=decision_config["name"],
            decision_window=decision_config.get("window"),
            persistence_k=decision_config.get("k"),
            persistence_n=decision_config.get("n")
        )
        
        self.plots.plot_metrics(results=predictions_history, attack_regions=attack_regions, title=title, window_size=window_evaluation, scenario_name=final_scenario_name, discretization_strategy=output_strategy_name)
        self.plots.plot_fp_fn(results=predictions_history, attack_regions=attack_regions, title=title, window_size=window_evaluation, scenario_name=final_scenario_name, discretization_strategy=output_strategy_name)
        self.plots.plot_score(results=predictions_history, attack_regions=attack_regions, title=title, discretization=final_discretization if final_discretization != 'params' else 0.5, scenario_name=final_scenario_name, discretization_strategy=output_strategy_name)