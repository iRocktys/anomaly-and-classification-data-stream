import optuna
import numpy as np
import time
import gc

from optuna import trial
from src.Anomaly.Models import get_anomaly_models
from src.Results.Metrics import Metrics
from src.Results.Plots import Plots

optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnomalyOptunaOptimizer:
    def __init__(self, stream, n_trials=30, discretization_threshold=0.5, target_names=None, n_runs=1):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.discretization_threshold = discretization_threshold
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

    def _optuna_callback(self, study, trial):
        f1, prec, rec = trial.user_attrs['metrics']
        params = trial.params
        print(f"Trial {trial.number + 1}/{self.n_trials} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Params: {params}")

    def _evaluate_model(self, model, threshold, warmup_instances, is_ae=False):
        self.stream.restart()
        y_true_list = []
        y_pred_list = []
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            score = model.score_instance(instance)
            
            if threshold == 'params':
                pred = model.predict(instance)
                predicted_class = 1 if (pred is not None and pred > 0) else 0
            else:
                predicted_class = 1 if score > threshold else 0
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(predicted_class)
            
            try:
                if not is_ae or predicted_class == 0:
                    model.train(instance)
            except ValueError:
                pass

        f1, prec, rec, mcc, fp, fn = self.metrics.calc_sklearn_metrics(y_true_list, y_pred_list)
        return f1, prec, rec

    def _run_trial_with_seeds(self, model_name, params, trial_threshold, warmup_instances, is_ae=False, n_seeds=1, early_stop_threshold=40.0):
        f1_list, prec_list, rec_list = [], [], []
        
        model_kwargs = params.copy()
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
                
            f1, prec, rec = self._evaluate_model(model, trial_threshold, warmup_instances, is_ae=is_ae)
            f1_list.append(f1)
            prec_list.append(prec)
            rec_list.append(rec)
            
            del model
            del models
            gc.collect()
            
            if i == 0 and f1 < early_stop_threshold:
                return f1, prec, rec
            
        return float(np.mean(f1_list)), float(np.mean(prec_list)), float(np.mean(rec_list))

    def _run_and_print_best_model(self, model_name, best_trial, warmup_instances, experiment_name, num_features, exec_id, window_evaluation):
        is_ae = any(kw.upper() in model_name.upper() for kw in ['AE', 'AUTOENCODER'])
        
        if self.discretization_threshold == 'dinamic':
            threshold = best_trial.params.get('dynamic_threshold', 0.5)
        elif self.discretization_threshold == 'params':
            if model_name == 'AIF':
                threshold = 0.5
            else:
                threshold = 'params'
        else:
            threshold = float(self.discretization_threshold)

        final_params = best_trial.params.copy()
        if 'dynamic_threshold' in final_params:
            del final_params['dynamic_threshold']
            
        model_kwargs = final_params.copy()
        if 'threshold' in model_kwargs and model_name in ['HST', 'OIF']:
            model_kwargs['anomaly_threshold'] = model_kwargs.pop('threshold')
        
        runs_data = []
        exec_times = []
        true_labels_multi = None
        actual_algo_name = model_name
        
        print(f"\n[{model_name}] Validando melhores parâmetros com {self.n_runs} rodada(s) prequencial(is)...")
        
        for run in range(self.n_runs):
            self.stream.restart()
            current_seed = 42 + run
            
            if model_name == 'HST':
                models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=model_kwargs, run_seed=current_seed)
                model = models['HalfSpaceTrees']
                actual_algo_name = 'HalfSpaceTrees'
            elif model_name == 'AIF':
                models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=model_kwargs, run_seed=current_seed)
                model = models['AdaptiveIsolationForest']
                actual_algo_name = 'AdaptiveIsolationForest'
            elif model_name == 'AE':
                models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=model_kwargs, run_seed=current_seed)
                model = models['Autoencoder']
                actual_algo_name = 'Autoencoder'
            elif model_name == 'OIF':
                models = get_anomaly_models(self.schema, selected_models=['OIF'], oif_params=model_kwargs, run_seed=current_seed)
                model = models['OnlineIsolationForest']
                actual_algo_name = 'OnlineIsolationForest'
            else:
                raise ValueError("Modelo não suportado.")

            y_true_list, y_pred_list, current_true_multi, scores = [], [], [], []
            instances_list, f1_list, prec_list, rec_list = [], [], [], []
            fp_list, fn_list = [], []
            
            count = 0
            start_time = time.time()
            
            while self.stream.has_more_instances():
                instance = self.stream.next_instance()
                true_label_multiclass = instance.y_index
                binary_true_label = 1 if true_label_multiclass > 0 else 0
                
                score = model.score_instance(instance)
                
                if threshold == 'params':
                    pred = model.predict(instance)
                    predicted_class = 1 if (pred is not None and pred > 0) else 0
                else:
                    predicted_class = 1 if score > threshold else 0
                
                scores.append(score)
                y_true_list.append(binary_true_label)
                y_pred_list.append(predicted_class)
                current_true_multi.append(true_label_multiclass)
                
                try:
                    if not is_ae or predicted_class == 0:
                        model.train(instance)
                except ValueError:
                    pass

                if window_evaluation and count >= warmup_instances and count > 0 and count % window_evaluation == 0:
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

            exec_time = time.time() - start_time
            exec_times.append(exec_time)
            
            if run == 0:
                true_labels_multi = current_true_multi
                
            runs_data.append({
                'y_true': y_true_list,
                'y_pred': y_pred_list,
                'true_labels_multi': true_labels_multi,
                'scores': scores,
                'instances': instances_list,
                'f1': f1_list,
                'precision': prec_list,
                'recall': rec_list,
                'fp': fp_list,
                'fn': fn_list
            })
            
            del model
            del models
            gc.collect()

        f1_matrix = np.array([r['f1'] for r in runs_data])
        prec_matrix = np.array([r['precision'] for r in runs_data])
        rec_matrix = np.array([r['recall'] for r in runs_data])
        fp_matrix = np.array([r['fp'] for r in runs_data])
        fn_matrix = np.array([r['fn'] for r in runs_data])
        scores_matrix = np.array([r['scores'] for r in runs_data])
        
        cum_metrics_list = []
        for r in runs_data:
            y_t = np.array(r['y_true'])[warmup_instances:] if len(r['y_true']) > warmup_instances else np.array(r['y_true'])
            y_p = np.array(r['y_pred'])[warmup_instances:] if len(r['y_pred']) > warmup_instances else np.array(r['y_pred'])
            cum_metrics_list.append(self.metrics.calc_sklearn_metrics(y_t, y_p))

        cum_matrix = np.array(cum_metrics_list)

        predictions_history = {
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
                'true_labels_multi': true_labels_multi
            }
        }
        
        feat_type = "FullFeatures" if (num_features is None or num_features > 50) else "33Features"
        scenario_name = f"Otimizado_{feat_type}"
        
        display_threshold = best_trial.params.get('threshold', 0.5) if threshold == 'params' else threshold
        
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history,
            warmup_instances=warmup_instances,
            n_runs=self.n_runs,
            params_dict=best_trial.params,
            experiment_name=experiment_name,
            scenario_name=scenario_name,
            discretization=display_threshold,
            window_evaluation=window_evaluation,
            exec_id=exec_id
        )

        attack_regions = self.metrics.extract_attack_regions(predictions_history[actual_algo_name]['true_labels_multi'], normal_class_idx=self.normal_class_idx)
        
        self.plots.plot_metrics(results=predictions_history, attack_regions=attack_regions, title=experiment_name, window_size=window_evaluation, scenario_name=scenario_name)
        self.plots.plot_fp_fn(results=predictions_history, attack_regions=attack_regions, title=experiment_name, window_size=window_evaluation, scenario_name=scenario_name)
        self.plots.plot_score(results=predictions_history, attack_regions=attack_regions, title=experiment_name, discretization=display_threshold, scenario_name=scenario_name)

    def optimize(self, model_name, warmup_instances=0, experiment_name="General", num_features=None, exec_id="N/A", window_evaluation=None):
        print(f"\n[{model_name}] Iniciando otimização focada no F1-Score (Binário | Média de Sementes com Early Stopping) com {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        is_ae = any(kw.upper() in model_name.upper() for kw in ['AE', 'AUTOENCODER'])
        
        def objective_wrapper(trial):
            if self.discretization_threshold == 'dinamic':
                trial_threshold = trial.suggest_float('dynamic_threshold', 0.05, 0.95)
            elif self.discretization_threshold == 'params':
                if model_name == 'AIF':
                    trial_threshold = 0.5
                else:
                    trial_threshold = 'params'
            else:
                trial_threshold = float(self.discretization_threshold)

            if model_name == 'HST':
                f1, prec, rec = self._objective_hst(trial, trial_threshold, warmup_instances)
            elif model_name == 'AIF':
                f1, prec, rec = self._objective_aif(trial, trial_threshold, warmup_instances)
            elif model_name == 'AE':
                f1, prec, rec = self._objective_ae(trial, trial_threshold, warmup_instances, is_ae)
            elif model_name == 'OIF':
                f1, prec, rec = self._objective_oif(trial, trial_threshold, warmup_instances)
            else:
                raise ValueError("Modelo não suportado.")
            
            trial.set_user_attr('metrics', (f1, prec, rec))
            gc.collect()
            try:
                import jpype
                if jpype.isJVMStarted():
                    jpype.java.lang.System.gc()
            except:
                pass
            return f1 

        study.optimize(objective_wrapper, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        
        print(f"\n[{model_name}] OTIMIZAÇÃO FINALIZADA")
        best_trial = study.best_trial
        best_f1, best_prec, best_rec = best_trial.user_attrs['metrics']
        
        print(f"Melhor Trial: {best_trial.number + 1}")
        print(f"Melhor Resultado -> F1: {best_f1:.4f} | Prec: {best_prec:.4f} | Rec: {best_rec:.4f}")
        print(f"Melhores Parâmetros: {study.best_params}")
        
        self.best_params[model_name] = study.best_params
        self._run_and_print_best_model(model_name, best_trial, warmup_instances, experiment_name, num_features, exec_id, window_evaluation)
        return study.best_params

    def _objective_hst(self, trial, trial_threshold, warmup_instances):
        params = {
            'window_size': trial.suggest_categorical('window_size', [500, 1000, 2048]),
            'number_of_trees': trial.suggest_int('number_of_trees', 30, 60, step=10),
            'max_depth': trial.suggest_int('max_depth', 12, 18),
            'size_limit': trial.suggest_float('size_limit', 0.01, 0.2)
        }
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
            
        return self._run_trial_with_seeds('HST', params, trial_threshold, warmup_instances)

    def _objective_aif(self, trial, trial_threshold, warmup_instances):
        params = {
            'window_size': trial.suggest_categorical('window_size', [128, 256, 512, 1024, 2048]),
            'height': trial.suggest_int('height', 5, 15),
            'weights': trial.suggest_float('weights', 0.0, 1.0)
        }
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
            
        return self._run_trial_with_seeds('AIF', params, trial_threshold, warmup_instances)

    def _objective_ae(self, trial, trial_threshold, warmup_instances, is_ae):
        params = {
            'hidden_layer': trial.suggest_categorical('hidden_layer', [4, 8, 16]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        }
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
            
        return self._run_trial_with_seeds('AE', params, trial_threshold, warmup_instances, is_ae=is_ae)
    
    def _objective_oif(self, trial, trial_threshold, warmup_instances):
        params = {
            'window_size': trial.suggest_categorical('window_size', [1024, 2048]),
            'num_trees': trial.suggest_int('num_trees', 30, 60, step=30),
            'max_leaf_samples': trial.suggest_categorical('max_leaf_samples', [16, 32]),
            'subsample': trial.suggest_float('subsample', 0.3, 0.6),
        }
        
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.1, 0.9)
            
        return self._run_trial_with_seeds('OIF', params, trial_threshold, warmup_instances)