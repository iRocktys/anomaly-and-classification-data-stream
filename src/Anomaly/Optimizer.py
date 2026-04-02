import optuna
import numpy as np
import time
import gc
from src.Anomaly.Models import get_anomaly_models
from src.Results.Metrics import Metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnomalyOptunaOptimizer:
    def __init__(self, stream, n_trials=30, discretization_threshold=0.5, target_class=None, target_names=None, n_runs=1):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.discretization_threshold = discretization_threshold
        self.target_class = target_class
        self.best_params = {}
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.metrics = Metrics()
        self.n_runs = n_runs

    def _optuna_callback(self, study, trial):
        f1, prec, rec = trial.user_attrs['metrics']
        params = trial.params
        print(f"Trial {trial.number + 1}/{self.n_trials} | F1: {f1:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | Params: {params}")

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

        f1, prec, rec, *_ = self.metrics.calc_sklearn_metrics(y_true_list, y_pred_list, target_class=self.target_class)
        return f1, prec, rec

    def _run_and_print_best_model(self, model_name, best_trial, warmup_instances):
        is_ae = any(kw.upper() in model_name.upper() for kw in ['AE', 'AUTOENCODER'])
        
        if self.discretization_threshold == 'dinamic':
            threshold = best_trial.params.get('dynamic_threshold', 0.5)
        elif self.discretization_threshold == 'params':
            threshold = 'params'
        else:
            threshold = float(self.discretization_threshold)

        final_params = best_trial.params.copy()
        if 'dynamic_threshold' in final_params:
            del final_params['dynamic_threshold']
        
        runs_cum_metrics = []
        exec_times = []
        true_labels_multi = None
        
        print(f"\n[{model_name}] Validando melhores parâmetros com {self.n_runs} rodada(s) prequencial(is)...")
        
        for run in range(self.n_runs):
            self.stream.restart()
            current_seed = 42 + run
            
            if model_name == 'HST':
                models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=final_params, run_seed=current_seed)
                model = models['HalfSpaceTrees']
            elif model_name == 'AIF':
                models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=final_params, run_seed=current_seed)
                model = models['AdaptiveIsolationForest']
            elif model_name == 'AE':
                models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=final_params, run_seed=current_seed)
                model = models['Autoencoder']
            else:
                raise ValueError("Modelo não suportado.")

            y_true_list = []
            y_pred_list = []
            current_true_multi = []
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
                
                y_true_list.append(binary_true_label)
                y_pred_list.append(predicted_class)
                current_true_multi.append(true_label_multiclass)
                
                try:
                    if not is_ae or predicted_class == 0:
                        model.train(instance)
                except ValueError:
                    pass

            exec_time = time.time() - start_time
            exec_times.append(exec_time)
            
            if run == 0:
                true_labels_multi = current_true_multi
                
            y_t = np.array(y_true_list)[warmup_instances:] if len(y_true_list) > warmup_instances else np.array(y_true_list)
            y_p = np.array(y_pred_list)[warmup_instances:] if len(y_pred_list) > warmup_instances else np.array(y_pred_list)
            
            cum_metrics = self.metrics.calc_sklearn_metrics(y_t, y_p, target_class=self.target_class)
            runs_cum_metrics.append(cum_metrics)
            
            del model
            del models
            gc.collect()

        cum_matrix = np.array(runs_cum_metrics)
        predictions_history = {
            f"Melhor {model_name}": {
                'exec_time_mean': np.mean(exec_times),
                'exec_time_std': np.std(exec_times) if self.n_runs > 1 else 0.0,
                'cumulative': {
                    'f1': (np.mean(cum_matrix[:, 0]), np.std(cum_matrix[:, 0]) if self.n_runs > 1 else 0.0),
                    'prec': (np.mean(cum_matrix[:, 1]), np.std(cum_matrix[:, 1]) if self.n_runs > 1 else 0.0),
                    'rec': (np.mean(cum_matrix[:, 2]), np.std(cum_matrix[:, 2]) if self.n_runs > 1 else 0.0),
                    'mcc': (np.mean(cum_matrix[:, 3]), np.std(cum_matrix[:, 3]) if self.n_runs > 1 else 0.0),
                    'fpr': (np.mean(cum_matrix[:, 4]), np.std(cum_matrix[:, 4]) if self.n_runs > 1 else 0.0),
                    'tpr': (np.mean(cum_matrix[:, 5]), np.std(cum_matrix[:, 5]) if self.n_runs > 1 else 0.0)
                },
                'true_labels_multi': true_labels_multi
            }
        }
        
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history,
            warmup_instances=warmup_instances,
            target_class=self.target_class,
            n_runs=self.n_runs
        )

    def optimize(self, model_name, warmup_instances=0):
        tgt_str = f"Classe {self.target_class}" if self.target_class is not None else "Macro"
        print(f"\n[{model_name}] Iniciando otimização focada no F1-Score ({tgt_str}) com {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        is_ae = any(kw.upper() in model_name.upper() for kw in ['AE', 'AUTOENCODER'])
        
        def objective_wrapper(trial):
            if self.discretization_threshold == 'dinamic':
                trial_threshold = trial.suggest_float('dynamic_threshold', 0.05, 0.95)
            elif self.discretization_threshold == 'params':
                trial_threshold = 'params'
            else:
                trial_threshold = float(self.discretization_threshold)

            if model_name == 'HST':
                f1, prec, rec = self._objective_hst(trial, trial_threshold, warmup_instances)
            elif model_name == 'AIF':
                f1, prec, rec = self._objective_aif(trial, trial_threshold, warmup_instances)
            elif model_name == 'AE':
                f1, prec, rec = self._objective_ae(trial, trial_threshold, warmup_instances, is_ae)
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
        print(f"Melhor Resultado -> F1: {best_f1:.2f} | Prec: {best_prec:.2f} | Rec: {best_rec:.2f}")
        print(f"Melhores Parâmetros: {study.best_params}")
        
        self.best_params[model_name] = study.best_params
        self._run_and_print_best_model(model_name, best_trial, warmup_instances)
        return study.best_params

    def _objective_hst(self, trial, trial_threshold, warmup_instances):
        params = {
            'window_size': trial.suggest_categorical('window_size', [256, 512, 1024, 2048]),
            'number_of_trees': trial.suggest_int('number_of_trees', 25, 100, step=25),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'size_limit': trial.suggest_float('size_limit', 0.01, 1.0)
        }
        if trial_threshold == 'params':
            params['anomaly_threshold'] = trial.suggest_float('anomaly_threshold', 0.05, 0.95)
            
        models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=params, run_seed=42)
        return self._evaluate_model(models['HalfSpaceTrees'], trial_threshold, warmup_instances)

    def _objective_aif(self, trial, trial_threshold, warmup_instances):
        params = {
            'window_size': trial.suggest_categorical('window_size', [256, 512, 1024, 2048]),
            'n_trees': trial.suggest_int('n_trees', 25, 100, step=25),
            'height': trial.suggest_int('height', 5, 15),
            'm_trees': trial.suggest_int('m_trees', 5, 50, step=5),
            'weights': trial.suggest_float('weights', 0.0, 1.0)
        }
        models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=params, run_seed=42)
        return self._evaluate_model(models['AdaptiveIsolationForest'], trial_threshold, warmup_instances)

    def _objective_ae(self, trial, trial_threshold, warmup_instances, is_ae):
        params = {
            'hidden_layer': trial.suggest_categorical('hidden_layer', [8, 16, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        }
        if trial_threshold == 'params':
            params['threshold'] = trial.suggest_float('threshold', 0.05, 0.95)
            
        models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=params, run_seed=42)
        return self._evaluate_model(models['Autoencoder'], trial_threshold, warmup_instances, is_ae=is_ae)