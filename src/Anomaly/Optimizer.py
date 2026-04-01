import optuna
import numpy as np
import time
from src.Anomaly.Models import get_anomaly_models
from src.Results.Metrics import Metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

class AnomalyOptunaOptimizer:
    def __init__(self, stream, n_trials=30, discretization_threshold=0.5, target_class=None, target_class_pass=None, target_names=None, n_runs=1):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.discretization_threshold = discretization_threshold
        self.target_class = target_class
        self.target_class_pass = target_class_pass
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
        
        count = 0
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
            count += 1

        f1_val, prec_val, recall_val, *_ = self.metrics.calc_sklearn_metrics(y_true_list, y_pred_list, target_class=self.target_class)
        return f1_val, prec_val, recall_val

    def _run_and_print_best_model(self, model_name, best_trial, warmup_instances, recovery_window=1000):
        is_ae = any(kw.upper() in model_name.upper() for kw in ['AE', 'AUTOENCODER'])
        
        if self.discretization_threshold == 'dinamic':
            threshold = best_trial.params.get('threshold', 0.5)
        elif self.discretization_threshold == 'params':
            threshold = 'params'
        else:
            threshold = self.discretization_threshold

        final_params = best_trial.params.copy()
        if 'threshold' in final_params and model_name != 'AE':
            del final_params['threshold']
        
        # Estruturas para armazenar resultados de múltiplas rodadas
        runs_cum_metrics = []
        runs_beh_metrics = []
        exec_times = []
        true_labels_multi = None
        
        print(f"\n[{model_name}] Validando melhores parâmetros com {self.n_runs} rodada(s) prequencial(is)...")
        
        for run in range(self.n_runs):
            self.stream.restart()
            
            if model_name == 'HST':
                models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=final_params)
                model = models['HalfSpaceTrees']
            elif model_name == 'AIF':
                models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=final_params)
                model = models['AdaptiveIsolationForest']
            elif model_name == 'AE':
                models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=final_params)
                model = models['Autoencoder']
            else:
                raise ValueError("Modelo não suportado.")

            y_true_list = []
            y_pred_list = []
            current_true_multi = []
            
            start_time = time.time()
            
            count = 0
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
                count += 1

            exec_time = time.time() - start_time
            exec_times.append(exec_time)
            
            if run == 0:
                true_labels_multi = current_true_multi
                
            # Calcular métricas cumulativas 
            y_t = np.array(y_true_list)[warmup_instances:] if len(y_true_list) > warmup_instances else np.array(y_true_list)
            y_p = np.array(y_pred_list)[warmup_instances:] if len(y_pred_list) > warmup_instances else np.array(y_pred_list)
            
            cum_metrics = self.metrics.calc_sklearn_metrics(y_t, y_p, target_class=self.target_class)
            runs_cum_metrics.append(cum_metrics)
            
            # Calcular métricas comportamentais
            normal_class_idx = 0
            for i, name in enumerate(self.target_names):
                if str(name).strip().upper() in ['BENIGN', 'NORMAL', '0']:
                    normal_class_idx = i
                    break
                    
            attack_regions = self.metrics.extract_attack_regions(current_true_multi, normal_class_idx=normal_class_idx)
            beh_metrics = self.metrics.calc_behavioral_metrics(y_true_list, y_pred_list, attack_regions, recovery_window, warmup_instances, self.target_class_pass)
            runs_beh_metrics.append(beh_metrics)

        # Matriz de métricas cumulativas para extração de médias e desvios
        cum_matrix = np.array(runs_cum_metrics)
        
        # Função para agregação das métricas comportamentais
        def aggregate_behavioral(beh_list, n):
            if not beh_list or not beh_list[0]: return []
            n_atk = len(beh_list[0])
            agg = []
            for i in range(n_atk):
                passagens = [r[i]['passagem'] for r in beh_list]
                recuperacoes = [r[i]['recuperacao'] for r in beh_list]
                agg.append({
                    'ataque_idx': beh_list[0][i]['ataque_idx'],
                    'passagem': (np.mean(passagens), np.std(passagens) if n > 1 else 0.0),
                    'recuperacao': (np.mean(recuperacoes), np.std(recuperacoes) if n > 1 else 0.0)
                })
            return agg

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
                'behavioral': aggregate_behavioral(runs_beh_metrics, self.n_runs),
                'true_labels_multi': true_labels_multi
            }
        }
        
        self.metrics.display_cumulative_metrics(
            predictions_history=predictions_history,
            warmup_instances=warmup_instances,
            target_class=self.target_class,
            target_class_pass=self.target_class_pass,
            recovery_window=recovery_window,
            normal_class_idx=normal_class_idx
        )

    def optimize(self, model_name, warmup_instances=0, recovery_window=1000):
        tgt_str = f"Classe {self.target_class}" if self.target_class is not None else "Macro"
        print(f"\n[{model_name}] Iniciando otimização focada no F1-Score ({tgt_str}) com {self.n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        is_ae = any(kw.upper() in model_name.upper() for kw in ['AE', 'AUTOENCODER'])
        
        def objective_wrapper(trial):
            if self.discretization_threshold == 'dinamic':
                trial_threshold = trial.suggest_float('threshold', 0.05, 0.95)
            elif self.discretization_threshold == 'params':
                trial_threshold = 'params'
            else:
                trial_threshold = self.discretization_threshold

            if model_name == 'HST':
                f1, prec, rec = self._objective_hst(trial, trial_threshold, warmup_instances)
            elif model_name == 'AIF':
                f1, prec, rec = self._objective_aif(trial, trial_threshold, warmup_instances)
            elif model_name == 'AE':
                f1, prec, rec = self._objective_ae(trial, trial_threshold, warmup_instances, is_ae)
            else:
                raise ValueError("Modelo não suportado.")
            
            trial.set_user_attr('metrics', (f1, prec, rec))
            return f1 

        study.optimize(objective_wrapper, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        
        print(f"\n[{model_name}] OTIMIZAÇÃO FINALIZADA")
        best_trial = study.best_trial
        best_f1, best_prec, best_rec = best_trial.user_attrs['metrics']
        
        print(f"Melhor Trial: {best_trial.number + 1}")
        print(f"Melhor Resultado -> F1: {best_f1:.2f} | Prec: {best_prec:.2f} | Rec: {best_rec:.2f}")
        print(f"Melhores Parâmetros: {study.best_params}")
        
        self.best_params[model_name] = study.best_params
        self._run_and_print_best_model(model_name, best_trial, warmup_instances, recovery_window)
        return study.best_params

    def _objective_hst(self, trial, trial_threshold, warmup_instances):
        hst_params = {
            'window_size': trial.suggest_int('window_size', 128, 2048, step=128),
            'number_of_trees': trial.suggest_int('number_of_trees', 10, 200, step=10),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'anomaly_threshold': trial.suggest_float('anomaly_threshold', 0.05, 0.95),
            'size_limit': trial.suggest_float('size_limit', 0.01, 1.0)
        }
        models = get_anomaly_models(self.schema, selected_models=['HST'], hst_params=hst_params)
        return self._evaluate_model(models['HalfSpaceTrees'], trial_threshold, warmup_instances)

    def _objective_aif(self, trial, trial_threshold, warmup_instances):
        aif_params = {
            'window_size': trial.suggest_int('window_size', 128, 2048, step=128),
            'n_trees': trial.suggest_int('n_trees', 10, 200, step=10),
            'height': trial.suggest_int('height', 5, 30),
            'm_trees': trial.suggest_int('m_trees', 5, 50, step=5),
            'weights': trial.suggest_float('weights', 0.0, 1.0)
        }
        models = get_anomaly_models(self.schema, selected_models=['AIF'], aif_params=aif_params)
        return self._evaluate_model(models['AdaptiveIsolationForest'], trial_threshold, warmup_instances)

    def _objective_ae(self, trial, trial_threshold, warmup_instances, is_ae):
        ae_params = {
            'hidden_layer': trial.suggest_int('hidden_layer', 4, 64),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'threshold': trial.suggest_float('threshold', 0.05, 0.95)
        }
        models = get_anomaly_models(self.schema, selected_models=['AE'], ae_params=ae_params)
        return self._evaluate_model(models['Autoencoder'], trial_threshold, warmup_instances, is_ae=is_ae)