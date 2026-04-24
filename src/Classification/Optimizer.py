import optuna
import numpy as np
import time
import gc
from src.Classification.Models import get_classification_models
from src.Results.Metrics import Metrics
from src.Results.Plots import Plots

optuna.logging.set_verbosity(optuna.logging.WARNING)

class ClassificationOptunaOptimizer:
    def __init__(self, stream, n_trials=30, target_names=None, n_runs=1):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
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

    def _evaluate_model(self, model):
        self.stream.restart()
        y_true_list = []
        y_pred_list = []
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index 
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            prediction = model.predict(instance)
            prediction = 0 if prediction is None else prediction
            binary_prediction = 1 if prediction > 0 else 0
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(binary_prediction)
            model.train(instance)

        f1_val, prec_val, recall_val, mcc_val, fp_val, fn_val = self.metrics.calc_sklearn_metrics(y_true_list, y_pred_list)
        return f1_val, prec_val, recall_val

    def _run_and_print_best_model(self, model_name, best_trial, warmup_instances=0, experiment_name="General", num_features=None, exec_id="N/A", window_evaluation=None):
        final_params = best_trial.params.copy()
        runs_data = []
        exec_times = []
        true_labels_multi = None
        actual_algo_name = model_name
        
        print(f"\n[{model_name}] Validando melhores parâmetros com {self.n_runs} rodada(s) prequencial(is)...")
        
        for run in range(self.n_runs):
            self.stream.restart()
            current_seed = 42 + run
            
            if model_name == 'LB':
                models = get_classification_models(self.schema, selected_models=['LB'], lb_params=final_params, run_seed=current_seed)
                model = models['LeveragingBagging']
                actual_algo_name = 'LeveragingBagging'
            elif model_name == 'HAT':
                models = get_classification_models(self.schema, selected_models=['HAT'], hat_params=final_params, run_seed=current_seed)
                model = models['HoeffdingAdaptiveTree']
                actual_algo_name = 'HoeffdingAdaptiveTree'
            elif model_name == 'ARF':
                models = get_classification_models(self.schema, selected_models=['ARF'], arf_params=final_params, run_seed=current_seed)
                model = models['AdaptiveRandomForest']
                actual_algo_name = 'AdaptiveRandomForest'
            elif model_name == 'HT':
                models = get_classification_models(self.schema, selected_models=['HT'], ht_params=final_params, run_seed=current_seed)
                model = models['HoeffdingTree']
                actual_algo_name = 'HoeffdingTree'
            else:
                raise ValueError("Modelo não suportado.")

            y_true_list, y_pred_list, current_true_multi = [], [], []
            instances_list, f1_list, prec_list, rec_list = [], [], [], []
            fp_list, fn_list = [], []
            
            count = 0
            start_time = time.time()
            
            while self.stream.has_more_instances():
                instance = self.stream.next_instance()
                true_label_multiclass = instance.y_index 
                binary_true_label = 1 if true_label_multiclass > 0 else 0
                
                prediction = model.predict(instance)
                prediction = 0 if prediction is None else prediction
                binary_prediction = 1 if prediction > 0 else 0
                
                y_true_list.append(binary_true_label)
                y_pred_list.append(binary_prediction)
                current_true_multi.append(true_label_multiclass)
                
                model.train(instance)
                
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
        
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history,
            warmup_instances=warmup_instances,
            n_runs=self.n_runs,
            params_dict=best_trial.params,
            experiment_name=experiment_name,
            scenario_name=scenario_name,
            discretization="N/A",
            window_evaluation=window_evaluation,
            exec_id=exec_id,
            discretization_strategy="classifier",
            task_type="classification"
        )

        attack_regions = self.metrics.extract_attack_regions(predictions_history[actual_algo_name]['true_labels_multi'], normal_class_idx=self.normal_class_idx)
        
        self.plots.plot_metrics(
            results=predictions_history, 
            attack_regions=attack_regions, 
            title=experiment_name, 
            window_size=window_evaluation,
            scenario_name=scenario_name,
            discretization_strategy="classifier"
        )
        
        self.plots.plot_fp_fn(
            results=predictions_history, 
            attack_regions=attack_regions, 
            title=experiment_name, 
            window_size=window_evaluation,
            scenario_name=scenario_name,
            discretization_strategy="classifier"
        )

    def optimize(self, model_name, warmup_instances=0, experiment_name="General", num_features=None, exec_id="N/A", window_evaluation=None):
        print(f"\n[{model_name}] Iniciando otimização focada no F1-Score (Binário) com {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        
        def objective_wrapper(trial):
            if model_name == 'LB':
                f1, prec, rec = self._objective_lb(trial)
            elif model_name == 'HAT':
                f1, prec, rec = self._objective_hat(trial)
            elif model_name == 'ARF':
                f1, prec, rec = self._objective_arf(trial)
            elif model_name == 'HT':
                f1, prec, rec = self._objective_ht(trial)
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
    
    def _objective_lb(self, trial):
        params = {
            'ensemble_size': trial.suggest_int('ensemble_size', 10, 100, step=10)
        }
        models = get_classification_models(self.schema, selected_models=['LB'], lb_params=params, run_seed=42)
        return self._evaluate_model(models['LeveragingBagging'])

    def _objective_hat(self, trial):
        params = {
            'grace_period': trial.suggest_int('grace_period', 10, 200, step=10),
            # 'split_criterion': trial.suggest_categorical('split_criterion', ['InfoGainSplitCriterion', 'GiniSplitCriterion']),
            'confidence': trial.suggest_float('confidence', 1e-5, 1e-1, log=True),
            'tie_threshold': trial.suggest_float('tie_threshold', 0.01, 0.1)
        }
        models = get_classification_models(self.schema, selected_models=['HAT'], hat_params=params, run_seed=42)
        return self._evaluate_model(models['HoeffdingAdaptiveTree']) 

    def _objective_arf(self, trial):
        params = {
            'ensemble_size': trial.suggest_int('ensemble_size', 10, 100, step=10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0, step=0.1),
            'lambda_param': trial.suggest_float('lambda_param', 1.0, 6.0, step=1.0)
        }
        models = get_classification_models(self.schema, selected_models=['ARF'], arf_params=params, run_seed=42)
        return self._evaluate_model(models['AdaptiveRandomForest']) 

    def _objective_ht(self, trial):
        params = {
            'grace_period': trial.suggest_int('grace_period', 10, 200, step=10),
            # 'split_criterion': trial.suggest_categorical('split_criterion', ['InfoGainSplitCriterion', 'GiniSplitCriterion', 'HellingerDistanceCriterion']),
            'confidence': trial.suggest_float('confidence', 1e-5, 1e-1, log=True),
            'tie_threshold': trial.suggest_float('tie_threshold', 0.01, 0.1)
        }
        models = get_classification_models(self.schema, selected_models=['HT'], ht_params=params, run_seed=42)
        return self._evaluate_model(models['HoeffdingTree'])