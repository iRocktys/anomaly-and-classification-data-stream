import optuna
import numpy as np
import time
import gc
from capymoa.evaluation import ClassificationEvaluator
from src.Anomaly.Models import get_anomaly_models
from src.Anomaly.Results import Metrics
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import math
from collections import deque

optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnomalyOptunaOptimizer:
    def __init__(self, stream, n_trials=30, discretization_threshold=0.5, target_class=None, target_class_pass=None, target_names=None):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.discretization_threshold = discretization_threshold
        self.target_class = target_class
        self.target_class_pass = target_class_pass
        self.best_params = {}
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.metrics = Metrics()

    def _optuna_callback(self, study, trial):
        f1, prec, rec = trial.user_attrs['metrics']
        params = trial.params
        print(f"Trial {trial.number + 1}/{self.n_trials} | "
              f"F1: {f1:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | "
              f"Params: {params}")

    def _evaluate_ae(self, model, trial_threshold, warmup_instances=0):
        self.stream.restart()
        
        y_true_list = []
        y_pred_list = []
        instance_idx = 0 
        min_warmup_required = max(warmup_instances, 0)

        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index 
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            is_warmup_phase = instance_idx < min_warmup_required
            raw_score = model.score_instance(instance) 
            predicted_class = 1 if raw_score > trial_threshold else 0
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(predicted_class)

            try:
                if is_warmup_phase or predicted_class == 0:
                    model.train(instance)
            except ValueError:
                pass
            instance_idx += 1

        eval_start = min_warmup_required
        y_true_eval = y_true_list[eval_start:] if len(y_true_list) > eval_start else y_true_list
        y_pred_eval = y_pred_list[eval_start:] if len(y_pred_list) > eval_start else y_pred_list

        f1_val, prec_val, recall_val = self._get_metric_classifier(y_true_eval, y_pred_eval, self.target_class)
        return f1_val, prec_val, recall_val
    
    def _evaluate_model(self, model, trial_threshold, warmup_instances=0):
        self.stream.restart()
        
        y_true_list = []
        y_pred_list = []
        instance_idx = 0
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index 
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            score = model.score_instance(instance) 
            predicted_class = 1 if score > trial_threshold else 0
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(predicted_class)

            try: 
                model.train(instance)
            except ValueError:
                pass
            
            instance_idx += 1

        y_true_eval = y_true_list[warmup_instances:] if len(y_true_list) > warmup_instances else y_true_list
        y_pred_eval = y_pred_list[warmup_instances:] if len(y_pred_list) > warmup_instances else y_pred_list

        f1_val, prec_val, recall_val = self._get_metric_classifier(y_true_eval, y_pred_eval, self.target_class)
        return f1_val, prec_val, recall_val

    def _run_and_print_best_model(self, model_name, best_trial, warmup_instances, recovery_window=1000):
        print(f"\n[RUN FINAL] Reconstruindo o melhor {model_name} para extração do relatório completo...")
        
        clean_params = best_trial.params.copy()
        trial_threshold = best_trial.user_attrs['trial_threshold']
        model_threshold = clean_params.get('threshold', clean_params.get('anomaly_threshold', trial_threshold))
        
        clean_params.pop('dynamic_threshold', None)
        
        is_ae = False
        if model_name == 'HST':
            p = clean_params.copy()
            p['anomaly_threshold'] = model_threshold
            models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=p)
            model = models['HalfSpaceTrees']
            warmup_instances = p.get('window_size', warmup_instances)
        elif model_name == 'OIF':
            models = get_anomaly_models(self.schema, selected_models=['OIF'], oif_params=clean_params)
            model = models['OnlineIsolationForest']
        elif model_name == 'AE':
            p = clean_params.copy()
            p['threshold'] = model_threshold
            models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=p)
            model = models['Autoencoder']
            is_ae = True
        elif model_name == 'RRCF':
            models = get_anomaly_models(self.schema, selected_models=['RRCF'], rrcf_params=clean_params)
            model = models['RobustRandomCutForest']
        elif model_name == 'AIF':
            models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=clean_params)
            model = models['AdaptiveIsolationForest']
        
        self.stream.restart()
        y_true_list = []
        y_true_multi_list = []
        y_pred_list = []
        
        start_time = time.time()
        instance_idx = 0
        min_warmup_required = max(warmup_instances, 0)
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index 
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            score = model.score_instance(instance) 
            predicted_class = 1 if score > trial_threshold else 0
            
            y_true_list.append(binary_true_label)
            y_true_multi_list.append(true_label_multiclass)
            y_pred_list.append(predicted_class)

            try: 
                if is_ae:
                    if instance_idx < min_warmup_required or predicted_class == 0:
                        model.train(instance)
                else:
                    model.train(instance)
            except ValueError:
                pass
                
            instance_idx += 1
            
        exec_time = time.time() - start_time
        
        predictions_history = {
            f"Melhor {model_name}": {
                'true_labels': y_true_list,
                'true_labels_multi': y_true_multi_list,
                'predicted_classes': y_pred_list,
                'exec_time': exec_time
            }
        }
        
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history, 
            warmup_instances=min_warmup_required, 
            target_class=self.target_class,
            target_class_pass=self.target_class_pass,
            recovery_window=recovery_window
        )

    def optimize(self, model_name, warmup_instances=0, recovery_window=1000):
        if self.target_class is None:
            tgt_str = "Híbrido (Macro Global)"
        elif str(self.target_class).lower() == 'macro':
            tgt_str = "Macro Total"
        else:
            tgt_str = f"Classe {self.target_class}"
            
        print(f"\n[{model_name}] Iniciando otimização focada no F1-Score ({tgt_str}) com {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        
        def objective_wrapper(trial):
            mode = str(self.discretization_threshold).lower()
            
            if mode in ["dinamico", "dinâmico"]:
                trial_threshold = trial.suggest_float('dynamic_threshold', 0.5, 0.95, step=0.05)
                if model_name == 'AE':
                    model_threshold = trial.suggest_float('threshold', 0.05, 0.9, step=0.05)
                elif model_name == 'HST':
                    model_threshold = trial.suggest_float('anomaly_threshold', 0.05, 0.95, step=0.05)
            elif mode == "params":
                if model_name == 'AE':
                    shared_threshold = trial.suggest_float('threshold', 0.1, 0.9, step=0.05)
                    trial_threshold = shared_threshold
                    model_threshold = shared_threshold
                elif model_name == 'HST':
                    shared_threshold = trial.suggest_float('anomaly_threshold', 0.05, 0.95, step=0.05)
                    trial_threshold = shared_threshold
                    model_threshold = shared_threshold
                else:
                    trial_threshold = trial.suggest_float('dynamic_threshold', 0.05, 0.95, step=0.05)
            else:
                trial_threshold = float(self.discretization_threshold)
                if model_name == 'AE':
                    model_threshold = trial.suggest_float('threshold', 0.05, 0.9, step=0.05)
                elif model_name == 'HST':
                    model_threshold = trial.suggest_float('anomaly_threshold', 0.05, 0.9, step=0.05)

            if model_name == 'HST':
                f1, prec, rec = self._objective_hst(trial, trial_threshold, model_threshold, warmup_instances)
            elif model_name == 'OIF':
                f1, prec, rec = self._objective_oif(trial, trial_threshold, warmup_instances)
            elif model_name == 'AE':
                f1, prec, rec = self._objective_ae(trial, trial_threshold, model_threshold, warmup_instances)
            elif model_name == 'RRCF':
                f1, prec, rec = self._objective_rrcf(trial, trial_threshold, warmup_instances)
            elif model_name == 'AIF':
                f1, prec, rec = self._objective_aif(trial, trial_threshold, warmup_instances)
            else:
                raise ValueError("Modelo não suportado.")
            
            trial.set_user_attr('metrics', (f1, prec, rec))
            trial.set_user_attr('model_name', model_name)
            trial.set_user_attr('trial_threshold', trial_threshold)
            
            gc.collect()
            
            return f1 

        study.optimize(objective_wrapper, n_trials=self.n_trials, callbacks=[self._optuna_callback], gc_after_trial=True)
        
        print(f"\n[{model_name}] OTIMIZAÇÃO FINALIZADA")
        best_trial = study.best_trial
        best_f1, best_prec, best_rec = best_trial.user_attrs['metrics']
        
        print(f"Melhor Trial: {best_trial.number + 1}")
        print(f"Melhor Resultado -> F1: {best_f1:.2f} | Prec: {best_prec:.2f} | Rec: {best_rec:.2f}")
        print(f"Melhores Parâmetros: {study.best_params}")
        
        self.best_params[model_name] = study.best_params
        
        self._run_and_print_best_model(model_name, best_trial, warmup_instances, recovery_window)
        
        return study.best_params

    def _objective_hst(self, trial, trial_threshold, model_threshold, warmup_instances):
        hst_params = {
            'window_size': trial.suggest_int('window_size', 100, 2050, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'anomaly_threshold': model_threshold,
            'size_limit': trial.suggest_float('size_limit', 0.05, 0.5, step=0.05)
        }
        models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=hst_params)
        dynamic_warmup = hst_params['window_size']
        res = self._evaluate_model(models['HalfSpaceTrees'], trial_threshold, warmup_instances=dynamic_warmup)
        del models
        return res

    def _objective_oif(self, trial, trial_threshold, warmup_instances):
        oif_params = {
            'num_trees': trial.suggest_int('num_trees', 10, 100, step=10),
            'max_leaf_samples': trial.suggest_int('max_leaf_samples', 16, 128, step=16),
            'growth_criterion': trial.suggest_categorical('growth_criterion', ['fixed', 'adaptive']),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.1),
            'window_size': trial.suggest_int('window_size', 512, 4096, step=256),
            'branching_factor': trial.suggest_int('branching_factor', 2, 5)
        }
        models = get_anomaly_models(self.schema, selected_models=['OIF'], oif_params=oif_params)
        res = self._evaluate_model(models['OnlineIsolationForest'], trial_threshold, warmup_instances=warmup_instances)
        del models
        return res

    def _objective_ae(self, trial, trial_threshold, model_threshold, warmup_instances):
        num_features = self.schema.get_num_attributes()
        max_hidden = max(1, num_features - 1)
        ae_params = {
            'hidden_layer': trial.suggest_int('hidden_layer', 1, max_hidden),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'threshold': model_threshold
        }
        models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=ae_params)
        res = self._evaluate_ae(models['Autoencoder'], trial_threshold, warmup_instances=warmup_instances)
        del models
        return res
    
    def _objective_rrcf(self, trial, trial_threshold, warmup_instances):
        rrcf_params = {
            'tree_size': trial.suggest_int('tree_size', 100, 2000, step=100),
            'n_trees': trial.suggest_int('n_trees', 10, 200, step=10)
        }
        models = get_anomaly_models(self.schema, selected_models=['RRCF'], rrcf_params=rrcf_params)
        res = self._evaluate_model(models['RobustRandomCutForest'], trial_threshold, warmup_instances=warmup_instances)
        del models
        return res

    def _objective_aif(self, trial, trial_threshold, warmup_instances):
        aif_params = {
            'window_size': trial.suggest_int('window_size', 128, 1024, step=128),
            'height': trial.suggest_int('height', 5, 20),
        }
        models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=aif_params)
        res = self._evaluate_model(models['AdaptiveIsolationForest'], trial_threshold, warmup_instances=warmup_instances)
        del models
        return res
        
    def _get_metric_classifier(self, y_true, y_pred, target_class=None):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0

        if target_class is None or str(target_class).lower() == 'macro':
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            tc = int(target_class)
            f1 = f1_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=tc, average='binary', zero_division=0)
            
        return f1 * 100.0, prec * 100.0, rec * 100.0