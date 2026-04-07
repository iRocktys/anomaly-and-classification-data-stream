import optuna
import numpy as np
import time
import gc
from src.Classification.Models import get_classification_models
from src.Results.Metrics import Metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

class ClassificationOptunaOptimizer:
    def __init__(self, stream, n_trials=30, target_class=1, target_names=None, n_runs=1):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.target_class = target_class
        self.best_params = {}
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']
        self.metrics = Metrics()
        self.n_runs = n_runs

    def _optuna_callback(self, study, trial):
        f1, prec, rec = trial.user_attrs['metrics']
        params = trial.params
        print(f"Trial {trial.number + 1}/{self.n_trials} | F1: {f1:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | Params: {params}")

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

        f1_val, prec_val, recall_val, *_ = self.metrics.calc_sklearn_metrics(y_true_list, y_pred_list, target_class=self.target_class)
        return f1_val, prec_val, recall_val

    def _run_and_print_best_model(self, model_name, best_trial, warmup_instances=0):
        final_params = best_trial.params.copy()
        runs_cum_metrics = []
        exec_times = []
        true_labels_multi = None
        
        print(f"\n[{model_name}] Validando melhores parâmetros com {self.n_runs} rodada(s) prequencial(is)...")
        
        for run in range(self.n_runs):
            self.stream.restart()
            current_seed = 42 + run
            
            if model_name == 'LB':
                models = get_classification_models(self.schema, selected_models=['LB'], lb_params=final_params, run_seed=current_seed)
                model = models['LeveragingBagging']
            elif model_name == 'HAT':
                models = get_classification_models(self.schema, selected_models=['HAT'], hat_params=final_params, run_seed=current_seed)
                model = models['HoeffdingAdaptiveTree']
            elif model_name == 'ARF':
                models = get_classification_models(self.schema, selected_models=['ARF'], arf_params=final_params, run_seed=current_seed)
                model = models['AdaptiveRandomForest']
            elif model_name == 'HT':
                models = get_classification_models(self.schema, selected_models=['HT'], ht_params=final_params, run_seed=current_seed)
                model = models['HoeffdingTree']
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
                
                prediction = model.predict(instance)
                prediction = 0 if prediction is None else prediction
                binary_prediction = 1 if prediction > 0 else 0
                
                y_true_list.append(binary_true_label)
                y_pred_list.append(binary_prediction)
                current_true_multi.append(true_label_multiclass)
                
                model.train(instance)

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
        print(f"Melhor Resultado -> F1: {best_f1:.2f} | Prec: {best_prec:.2f} | Rec: {best_rec:.2f}")
        print(f"Melhores Parâmetros: {study.best_params}")
        # Separa as chaves por tabulação
        chaves = "\t".join(study.best_params.keys())

        # Separa os valores por tabulação e substitui ponto por vírgula
        valores = "\t".join([str(v).replace('.', ',') for v in study.best_params.values()])

        print(f"Melhores Parâmetros:\n{chaves}\n{valores}")
        
        self.best_params[model_name] = study.best_params
        self._run_and_print_best_model(model_name, best_trial, warmup_instances)
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
            'split_criterion': trial.suggest_categorical('split_criterion', ['InfoGainSplitCriterion', 'GiniSplitCriterion']),
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
            'split_criterion': trial.suggest_categorical('split_criterion', ['InfoGainSplitCriterion', 'GiniSplitCriterion']),
            'confidence': trial.suggest_float('confidence', 1e-5, 1e-1, log=True),
            'tie_threshold': trial.suggest_float('tie_threshold', 0.01, 0.1)
        }
        models = get_classification_models(self.schema, selected_models=['HT'], ht_params=params, run_seed=42)
        return self._evaluate_model(models['HoeffdingTree'])