import optuna
import numpy as np
import time
import gc
import os

from optuna import trial
from torchgen import model
from src.Anomaly.Models import get_anomaly_models
from src.Results.Metrics import Metrics
from src.Results.Plots import Plots

optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnomalyOptunaOptimizer:
    def __init__(self, stream, n_trials=30, discretization_threshold=0.5, target_names=None, n_runs=1, decision_strategy="raw", fp_penalty_weight=0.5):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.discretization_threshold = discretization_threshold
        self.decision_strategy = decision_strategy
        self.decision_config = self._parse_decision_strategy(decision_strategy)
        # Peso da penalização da taxa de falsos positivos no objetivo do Optuna.
        # Objetivo = F1 - fp_penalty_weight * FPR(%).
        self.fp_penalty_weight = float(fp_penalty_weight)
        self.best_params = {}
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        
        self.normal_class_idx = 0
        for i, name in enumerate(self.target_names):
            if str(name).strip().upper() in ['BENIGN', 'NORMAL', '0']:
                self.normal_class_idx = i
                break
                
        self.metrics = Metrics()
        self.plots = Plots(self.target_names)
        self.n_runs = n_runs

    def _parse_decision_strategy(self, decision_strategy):
        strategy = str(decision_strategy or "raw").strip().lower()

        if strategy == "raw":
            return {"name": "raw", "type": "raw", "window": None, "k": None, "n": None}

        if strategy.startswith("moving_average_w"):
            try:
                window = int(strategy.split("_w")[-1])
            except ValueError:
                raise ValueError(f"Estratégia de média móvel inválida: {decision_strategy}")
            if window <= 0:
                raise ValueError("A janela da média móvel precisa ser positiva.")
            return {"name": strategy, "type": "moving_average", "window": window, "k": None, "n": None}

        if strategy.startswith("persistence_") and "_of_" in strategy:
            try:
                left, right = strategy.replace("persistence_", "").split("_of_")
                k, n = int(left), int(right)
            except ValueError:
                raise ValueError(f"Estratégia de persistência inválida: {decision_strategy}")
            if k <= 0 or n <= 0 or k > n:
                raise ValueError("A persistência precisa obedecer 0 < k <= n.")
            return {"name": strategy, "type": "persistence", "window": None, "k": k, "n": n}

        raise ValueError(
            "decision_strategy deve ser uma destas: raw, moving_average_w3, "
            "moving_average_w5, persistence_2_of_3, persistence_3_of_5."
        )

    def _combined_strategy_name(self, threshold_strategy):
        return os.path.join(str(threshold_strategy), self.decision_config["name"])

    def _causal_moving_average(self, values, window):
        if not values:
            return 0.0
        window = max(1, int(window))
        recent = values[-window:]
        return float(np.mean(recent))

    def _build_causal_moving_average_series(self, values, window):
        values = list(values)
        if not values:
            return []
        return [float(np.mean(values[max(0, i - window + 1):i + 1])) for i in range(len(values))]

    def _compute_z_threshold(self, warmup_scores, z_value):
        if self.decision_config["type"] == "moving_average":
            reference_scores = self._build_causal_moving_average_series(warmup_scores, self.decision_config["window"])
        else:
            reference_scores = list(warmup_scores)

        if not reference_scores:
            return 0.5, None, None

        mu = float(np.mean(reference_scores))
        std = float(np.std(reference_scores))
        threshold = mu + (float(z_value) * std)
        return threshold, mu, std

    def _apply_decision_rule(self, score, threshold, score_history, peak_history):
        decision_type = self.decision_config["type"]

        if decision_type == "moving_average":
            decision_score = self._causal_moving_average(score_history, self.decision_config["window"])
            return 1 if decision_score > threshold else 0

        if decision_type == "persistence":
            is_peak = 1 if score > threshold else 0
            peak_history.append(is_peak)
            recent = peak_history[-self.decision_config["n"]:]
            return 1 if sum(recent) >= self.decision_config["k"] else 0

        return 1 if score > threshold else 0

    def _fp_penalized_objective(self, f1, fp, y_true_eval):
        y_true_arr = np.asarray(y_true_eval)
        normal_count = int(np.sum(y_true_arr == 0))
        fpr = (float(fp) / normal_count * 100.0) if normal_count > 0 else 0.0
        objective_score = float(f1) - (self.fp_penalty_weight * fpr)
        return objective_score, fpr

    def optuna_callback(self, study, trial):
        metrics = trial.user_attrs.get('metrics', ())
        if len(metrics) >= 5:
            f1, prec, rec, objective_score, fpr = metrics[:5]
        else:
            f1, prec, rec = metrics[:3]
            objective_score, fpr = f1, 0.0
        params = trial.params
        print(
            f"Trial {trial.number + 1}/{self.n_trials} | "
            f"Obj: {objective_score:.4f} | F1: {f1:.4f} | "
            f"Prec: {prec:.4f} | Rec: {rec:.4f} | FPR: {fpr:.4f}% | "
            f"Params: {params}"
        )

    def evaluate_model(self, model, threshold_mode, warmup_instances, z_value=None, is_ae=False):
        self.stream.restart()
        y_true_list = []
        y_pred_list = []
        warmup_scores = []
        score_history = []
        peak_history = []

        threshold = threshold_mode if isinstance(threshold_mode, (float, int)) else 0.5
        count = 0

        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index
            binary_true_label = 1 if true_label_multiclass > 0 else 0

            score = model.score_instance(instance)
            score_history.append(score)

            if count < warmup_instances:
                warmup_scores.append(score)
                predicted_class = 0
                try:
                    model.train(instance)
                except ValueError:
                    pass
            else:
                if count == warmup_instances and z_value is not None and warmup_instances > 0:
                    threshold, _, _ = self._compute_z_threshold(warmup_scores, z_value)
                    threshold = min(threshold, 0.999)

                if threshold_mode == 'params':
                    pred = model.predict(instance)
                    predicted_class = 1 if (pred is not None and pred > 0) else 0
                else:
                    predicted_class = self._apply_decision_rule(score, threshold, score_history, peak_history)

                try:
                    if not is_ae or predicted_class == 0:
                        model.train(instance)
                except ValueError:
                    pass

            y_true_list.append(binary_true_label)
            y_pred_list.append(predicted_class)
            count += 1

        y_t_eval = y_true_list[warmup_instances:] if len(y_true_list) > warmup_instances else y_true_list
        y_p_eval = y_pred_list[warmup_instances:] if len(y_pred_list) > warmup_instances else y_pred_list

        f1, prec, rec, mcc, fp, fn = self.metrics.calc_sklearn_metrics(y_t_eval, y_p_eval)
        objective_score, fpr = self._fp_penalized_objective(f1, fp, y_t_eval)
        return objective_score, f1, prec, rec, fpr

    def run_trial_with_seeds(self, model_name, params, trial_threshold, warmup_instances, is_ae=False, n_seeds=1, early_stop_threshold=35.0):
        objective_list, f1_list, prec_list, rec_list, fpr_list = [], [], [], [], []

        model_kwargs = params.copy()
        z_value = model_kwargs.pop('z', None)

        if 'threshold' in model_kwargs and model_name in ['HST', 'OIF']:
            model_kwargs['anomaly_threshold'] = model_kwargs.pop('threshold')

        for i, seed in enumerate(range(42, 42 + n_seeds)):
            if model_name == 'HST':
                models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=model_kwargs, run_seed=seed)
                model = models['HalfSpaceTrees']
            elif model_name == 'AIF':
                models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=model_kwargs, run_seed=seed)
                model = models['AdaptiveIsolationForest']
            elif model_name == 'AE':
                models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=model_kwargs, run_seed=seed)
                model = models['Autoencoder']
            elif model_name == 'OIF':
                models = get_anomaly_models(self.schema, selected_models=['OIF'], oif_params=model_kwargs, run_seed=seed)
                model = models['OnlineIsolationForest']
            else:
                raise ValueError("Modelo não suportado na otimização com sementes.")

            objective_score, f1, prec, rec, fpr = self.evaluate_model(
                model, trial_threshold, warmup_instances, z_value=z_value, is_ae=is_ae
            )
            objective_list.append(objective_score)
            f1_list.append(f1)
            prec_list.append(prec)
            rec_list.append(rec)
            fpr_list.append(fpr)

            del model
            del models
            gc.collect()

            if i == 0 and objective_score < early_stop_threshold:
                return objective_score, f1, prec, rec, fpr

        objective_mean = float(np.mean(objective_list))
        objective_std = float(np.std(objective_list))
        objective_otimizado = objective_mean - objective_std

        return (
            objective_otimizado,
            float(np.mean(f1_list)),
            float(np.mean(prec_list)),
            float(np.mean(rec_list)),
            float(np.mean(fpr_list)),
        )

    def execute_single_run(self, model_name, model_kwargs, current_seed, warmup_instances, threshold_mode, z_value, window_evaluation, is_ae):
        self.stream.restart()
        
        if model_name == 'HST':
            models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=model_kwargs, run_seed=current_seed)
            model = models['HalfSpaceTrees']
        elif model_name == 'AIF':
            models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=model_kwargs, run_seed=current_seed)
            model = models['AdaptiveIsolationForest']
        elif model_name == 'AE':
            models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=model_kwargs, run_seed=current_seed)
            model = models['Autoencoder']
        elif model_name == 'OIF':
            models = get_anomaly_models(self.schema, selected_models=['OIF'], oif_params=model_kwargs, run_seed=current_seed)
            model = models['OnlineIsolationForest']
        else:
            raise ValueError("Modelo não suportado.")

        y_true_list, y_pred_list, current_true_multi, scores = [], [], [], []
        instances_list, f1_list, prec_list, rec_list = [], [], [], []
        fp_list, fn_list = [], []
        warmup_scores = []
        score_history = []
        peak_history = []
        
        run_threshold = threshold_mode if isinstance(threshold_mode, (float, int)) else 0.5
        count = 0
        calc_mu, calc_std = None, None
        start_time = time.time()
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            score = model.score_instance(instance)
            scores.append(score)
            score_history.append(score)
            current_true_multi.append(true_label_multiclass)
            
            if count < warmup_instances:
                warmup_scores.append(score)
                predicted_class = 0
                try:
                    model.train(instance)
                except ValueError:
                    pass
            else:
                if count == warmup_instances and z_value is not None and warmup_instances > 0:
                    run_threshold, calc_mu, calc_std = self._compute_z_threshold(warmup_scores, z_value)
                    run_threshold = min(run_threshold, 0.999)
                    
                if threshold_mode == 'params':
                    pred = model.predict(instance)
                    predicted_class = 1 if (pred is not None and pred > 0) else 0
                else:
                    predicted_class = self._apply_decision_rule(score, run_threshold, score_history, peak_history)
                
                try:
                    if not is_ae or predicted_class == 0:
                        model.train(instance)
                except ValueError:
                    pass

            y_true_list.append(binary_true_label)
            y_pred_list.append(predicted_class)

            if window_evaluation and count >= warmup_instances and count > 0 and count % window_evaluation == 0:
                start_idx = max(warmup_instances, len(y_true_list) - window_evaluation)
                y_t_win = y_true_list[start_idx:]
                y_p_win = y_pred_list[start_idx:]

                # Métricas janeladas: mostram a variação local de desempenho em cada bloco,
                # permitindo observar quedas/subidas nas regiões de ataque.
                f1_v, prec_v, rec_v, _, fp_v, fn_v = self.metrics.calc_sklearn_metrics(y_t_win, y_p_win)
                
                instances_list.append(count)
                f1_list.append(f1_v)
                prec_list.append(prec_v)
                rec_list.append(rec_v)
                fp_list.append(fp_v)
                fn_list.append(fn_v)
                
            count += 1

        exec_time = time.time() - start_time
        
        del model
        del models
        gc.collect()

        return {
            'y_true': y_true_list,
            'y_pred': y_pred_list,
            'true_labels_multi': current_true_multi,
            'scores': scores,
            'instances': instances_list,
            'f1': f1_list,
            'precision': prec_list,
            'recall': rec_list,
            'fp': fp_list,
            'fn': fn_list,
            'exec_time': exec_time,
            'z_stats': {'mu': calc_mu, 'std': calc_std, 'threshold': run_threshold if z_value is not None else None}
        }

    def compile_runs_history(self, runs_data, warmup_instances, actual_algo_name):
        f1_matrix = np.array([r['f1'] for r in runs_data])
        prec_matrix = np.array([r['precision'] for r in runs_data])
        rec_matrix = np.array([r['recall'] for r in runs_data])
        fp_matrix = np.array([r['fp'] for r in runs_data])
        fn_matrix = np.array([r['fn'] for r in runs_data])
        scores_matrix = np.array([r['scores'] for r in runs_data])
        exec_times = [r['exec_time'] for r in runs_data]
        
        cum_metrics_list = []
        for r in runs_data:
            y_t = np.array(r['y_true'])[warmup_instances:] if len(r['y_true']) > warmup_instances else np.array(r['y_true'])
            y_p = np.array(r['y_pred'])[warmup_instances:] if len(r['y_pred']) > warmup_instances else np.array(r['y_pred'])
            cum_metrics_list.append(self.metrics.calc_sklearn_metrics(y_t, y_p))

        cum_matrix = np.array(cum_metrics_list)

        return {
            actual_algo_name: {
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
                'true_labels_multi': runs_data[0]['true_labels_multi']
            }
        }

    def export_metrics_and_plots(self, predictions_history, actual_algo_name, warmup_instances, num_features, experiment_name, exec_id, window_evaluation, best_trial_params, z_value, mu_list, std_list, thresh_list, threshold_mode, strategy_name):
        feat_type = "FullFeatures" if (num_features is None or num_features > 50) else "33Features"
        scenario_name = f"Otimizado_{feat_type}"
        output_strategy_name = self._combined_strategy_name(strategy_name)
        
        display_params = best_trial_params.copy()
        display_params['decision_strategy'] = self.decision_config['name']
        display_params['fp_penalty_weight'] = self.fp_penalty_weight
        display_params['decision_type'] = self.decision_config['type']
        display_params['decision_window'] = self.decision_config.get('window')
        display_params['persistence_k'] = self.decision_config.get('k')
        display_params['persistence_n'] = self.decision_config.get('n')
        if z_value is not None:
            display_params['z'] = float(z_value)
            display_params['u'] = float(np.mean(mu_list)) if mu_list else None
            display_params['std'] = float(np.mean(std_list)) if std_list else None
            display_threshold = float(np.mean(thresh_list)) if thresh_list else None
        else:
            display_threshold = display_params.get('threshold', 0.5) if threshold_mode == 'params' else threshold_mode
        
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history,
            warmup_instances=warmup_instances,
            n_runs=self.n_runs,
            params_dict=display_params,
            experiment_name=experiment_name,
            scenario_name=scenario_name,
            discretization=display_threshold,
            window_evaluation=window_evaluation,
            exec_id=exec_id,
            discretization_strategy=output_strategy_name,
            task_type="anomaly",
            threshold_strategy=strategy_name,
            decision_strategy=self.decision_config['name'],
            decision_window=self.decision_config.get('window'),
            persistence_k=self.decision_config.get('k'),
            persistence_n=self.decision_config.get('n'),
        )

        attack_regions = self.metrics.extract_attack_regions(predictions_history[actual_algo_name]['true_labels_multi'], normal_class_idx=self.normal_class_idx)
        
        self.plots.plot_metrics(results=predictions_history, attack_regions=attack_regions, title=experiment_name, window_size=window_evaluation, scenario_name=scenario_name, discretization_strategy=output_strategy_name)
        self.plots.plot_fp_fn(results=predictions_history, attack_regions=attack_regions, title=experiment_name, window_size=window_evaluation, scenario_name=scenario_name, discretization_strategy=output_strategy_name)
        self.plots.plot_score(results=predictions_history, attack_regions=attack_regions, title=experiment_name, discretization=display_threshold, scenario_name=scenario_name, discretization_strategy=output_strategy_name)

    def validate_best_parameters(self, model_name, best_trial, warmup_instances, experiment_name, num_features, exec_id, window_evaluation):
        is_ae = any(kw.upper() in model_name.upper() for kw in ['AE', 'AUTOENCODER'])
        
        if self.discretization_threshold == 'dinamic':
            threshold_mode = best_trial.params.get('dynamic_threshold', 0.5)
            strategy_name = 'dinamic'
        elif self.discretization_threshold == 'params':
            threshold_mode = 0.5 if model_name == 'AIF' else 'params'
            strategy_name = 'params'
        elif self.discretization_threshold == 'z_score':
            threshold_mode = 'z_score'
            strategy_name = 'z_score'
        else:
            threshold_mode = float(self.discretization_threshold)
            strategy_name = 'fixed'

        final_params = best_trial.params.copy()
        z_value = final_params.get('z', None)
        
        if 'dynamic_threshold' in final_params:
            del final_params['dynamic_threshold']
        if 'z' in final_params:
            del final_params['z']
            
        model_kwargs = final_params.copy()
        if 'threshold' in model_kwargs and model_name in ['HST', 'OIF']:
            model_kwargs['anomaly_threshold'] = model_kwargs.pop('threshold')
            
        actual_algo_mapping = {'HST': 'HalfSpaceTrees', 'AIF': 'AdaptiveIsolationForest', 'AE': 'Autoencoder', 'OIF': 'OnlineIsolationForest'}
        actual_algo_name = actual_algo_mapping.get(model_name, model_name)
        
        runs_data = []
        mu_list, std_list, thresh_list = [], [], []
        
        print(f"\n[{model_name}] Validando melhores parâmetros com {self.n_runs} rodada(s) prequencial(is)...")
        
        for run in range(self.n_runs):
            current_seed = 42 + run
            run_result = self.execute_single_run(model_name, model_kwargs, current_seed, warmup_instances, threshold_mode, z_value, window_evaluation, is_ae)
            runs_data.append(run_result)
            
            if run_result['z_stats']['threshold'] is not None:
                mu_list.append(run_result['z_stats']['mu'])
                std_list.append(run_result['z_stats']['std'])
                thresh_list.append(run_result['z_stats']['threshold'])
                
        predictions_history = self.compile_runs_history(runs_data, warmup_instances, actual_algo_name)
        
        self.export_metrics_and_plots(
            predictions_history, actual_algo_name, warmup_instances, num_features,
            experiment_name, exec_id, window_evaluation, best_trial.params,
            z_value, mu_list, std_list, thresh_list, threshold_mode, strategy_name
        )

    def optimize(self, model_name, warmup_instances=0, experiment_name="General", num_features=None, exec_id="N/A", window_evaluation=None):
        print(f"\n[{model_name}] Iniciando otimização focada em F1 penalizado por FP (Binário | Decisão: {self.decision_config['name']} | Média de Sementes com Early Stopping) com {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        is_ae = any(kw.upper() in model_name.upper() for kw in ['AE', 'AUTOENCODER'])
        
        def objective_wrapper(trial):
            if self.discretization_threshold == 'dinamic':
                trial_threshold = trial.suggest_float('dynamic_threshold', 0.05, 0.95)
            elif self.discretization_threshold == 'z_score':
                trial_threshold = 'z_score'
            elif self.discretization_threshold == 'params':
                if model_name == 'AIF':
                    trial_threshold = 0.5
                else:
                    trial_threshold = 'params'
            else:
                trial_threshold = float(self.discretization_threshold)

            if model_name == 'HST':
                objective_score, f1, prec, rec, fpr = self.objective_hst(trial, trial_threshold, warmup_instances)
            elif model_name == 'AIF':
                objective_score, f1, prec, rec, fpr = self.objective_aif(trial, trial_threshold, warmup_instances)
            elif model_name == 'AE':
                objective_score, f1, prec, rec, fpr = self.objective_ae(trial, trial_threshold, warmup_instances, is_ae)
            elif model_name == 'OIF':
                objective_score, f1, prec, rec, fpr = self.objective_oif(trial, trial_threshold, warmup_instances)
            else:
                raise ValueError("Modelo não suportado.")

            trial.set_user_attr('metrics', (f1, prec, rec, objective_score, fpr))
            gc.collect()
            try:
                import jpype
                if jpype.isJVMStarted():
                    jpype.java.lang.System.gc()
            except:
                pass
            return objective_score 

        study.optimize(objective_wrapper, n_trials=self.n_trials, callbacks=[self.optuna_callback])
        
        print(f"\n[{model_name}] OTIMIZAÇÃO FINALIZADA")
        best_trial = study.best_trial
        best_metrics = best_trial.user_attrs['metrics']
        best_f1, best_prec, best_rec = best_metrics[:3]
        best_objective = best_metrics[3] if len(best_metrics) > 3 else best_f1
        best_fpr = best_metrics[4] if len(best_metrics) > 4 else 0.0

        print(f"Melhor Trial: {best_trial.number + 1}")
        print(
            f"Melhor Resultado -> Obj: {best_objective:.4f} | F1: {best_f1:.4f} | "
            f"Prec: {best_prec:.4f} | Rec: {best_rec:.4f} | FPR: {best_fpr:.4f}%"
        )
        print(f"Melhores Parâmetros: {study.best_params}")
        
        self.best_params[model_name] = study.best_params
        self.validate_best_parameters(model_name, best_trial, warmup_instances, experiment_name, num_features, exec_id, window_evaluation)
        return study.best_params

    def objective_hst(self, trial, trial_threshold, warmup_instances):
        params = {
            'window_size': trial.suggest_categorical('window_size', [128, 256, 512, 1024]),
            'number_of_trees': trial.suggest_int('number_of_trees', 30, 60, step=10),
            'max_depth': trial.suggest_int('max_depth', 12, 18),
            'size_limit': trial.suggest_float('size_limit', 0.01, 0.2)
        }
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
        elif trial_threshold == 'z_score':
            params['z'] = trial.suggest_float('z', 1.0, 10.0)
            
        return self.run_trial_with_seeds('HST', params, trial_threshold, warmup_instances, n_seeds=3)

    def objective_aif(self, trial, trial_threshold, warmup_instances):
        params = {
            'window_size': trial.suggest_categorical('window_size', [128, 256, 512, 1024]),
            'height': trial.suggest_int('height', 5, 15),
            'weights': trial.suggest_float('weights', 0.0, 1.0)
        }
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
        elif trial_threshold == 'z_score':
            params['z'] = trial.suggest_float('z', 1.0, 10.0)
            
        return self.run_trial_with_seeds('AIF', params, trial_threshold, warmup_instances, n_seeds=3)

    def objective_ae(self, trial, trial_threshold, warmup_instances, is_ae):
        params = {
            'hidden_layer': trial.suggest_categorical('hidden_layer', [4, 8, 16]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        }
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
        elif trial_threshold == 'z_score':
            params['z'] = trial.suggest_float('z', 1.0, 10.0)
            
        return self.run_trial_with_seeds('AE', params, trial_threshold, warmup_instances, is_ae=is_ae, n_seeds=3)
    
    def objective_oif(self, trial, trial_threshold, warmup_instances):
        params = {
            'window_size': trial.suggest_categorical('window_size', [128, 256, 512, 1024]),
            'num_trees': trial.suggest_int('num_trees', 30, 60, step=30),
            'max_leaf_samples': trial.suggest_categorical('max_leaf_samples', [8, 16, 32]),
            'subsample': trial.suggest_float('subsample', 0.3, 0.6),
        }
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
        elif trial_threshold == 'z_score':
            params['z'] = trial.suggest_float('z', 1.0, 10.0)
            
        return self.run_trial_with_seeds('OIF', params, trial_threshold, warmup_instances, n_seeds=3)