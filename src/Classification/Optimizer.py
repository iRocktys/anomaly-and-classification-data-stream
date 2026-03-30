import optuna
from sklearn.metrics import f1_score, precision_score, recall_score
from Models import get_classification_models

optuna.logging.set_verbosity(optuna.logging.WARNING)

class ClassificationOptunaOptimizer:
    def __init__(self, stream, n_trials=30, target_class=1):
        self.stream = stream
        self.schema = stream.get_schema()
        self.n_trials = n_trials
        self.target_class = target_class
        self.best_params = {}

    def _optuna_callback(self, study, trial):
        f1, prec, rec = trial.user_attrs['metrics']
        params = trial.params
        
        print(f"Trial {trial.number + 1}/{self.n_trials} | "
              f"F1: {f1:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | "
              f"Params: {params}")

    def _evaluate_model(self, model):
        self.stream.restart()
        
        y_true_list = []
        y_pred_list = []
        
        while self.stream.has_more_instances():
            instance = self.stream.next_instance()
            true_label_multiclass = instance.y_index 
            binary_true_label = 1 if true_label_multiclass > 0 else 0
            
            prediction = model.predict(instance)
            if prediction is None:
                prediction = 0
                
            binary_prediction = 1 if prediction > 0 else 0
            
            y_true_list.append(binary_true_label)
            y_pred_list.append(binary_prediction)

            model.train(instance)

        f1_val, prec_val, recall_val = self._get_metric_classifier(
            y_true=y_true_list, 
            y_pred=y_pred_list, 
            target_class=self.target_class
        )

        return f1_val, prec_val, recall_val

    def optimize(self, model_name):
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
            
            return f1 

        study.optimize(objective_wrapper, n_trials=self.n_trials, callbacks=[self._optuna_callback])
        
        print(f"\n[{model_name}] OTIMIZAÇÃO FINALIZADA")
        
        best_trial = study.best_trial
        best_f1, best_prec, best_rec = best_trial.user_attrs['metrics']
        
        print(f"Melhor Trial: {best_trial.number + 1}")
        print(f"Melhor Resultado -> F1: {best_f1:.2f} | Prec: {best_prec:.2f} | Rec: {best_rec:.2f}")
        print(f"Melhores Parâmetros: {study.best_params}")
        
        self.best_params[model_name] = study.best_params
        return study.best_params
    
    def _get_metric_classifier(self, y_true, y_pred, target_class=1):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0

        if target_class is None:
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            f1 = f1_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            
        return f1 * 100.0, prec * 100.0, rec * 100.0
    
    def _objective_lb(self, trial):
        lb_params = {
            'ensemble_size': trial.suggest_int('ensemble_size', 10, 150, step=10)
        }
        models = get_classification_models(self.schema, selected_models=['LB'], lb_params=lb_params)
        return self._evaluate_model(models['LeveragingBagging'])

    def _objective_hat(self, trial):
        hat_params = {
            'grace_period': trial.suggest_int('grace_period', 10, 500, step=10),
            'split_criterion': trial.suggest_categorical('split_criterion', ['InfoGainSplitCriterion', 'GiniSplitCriterion']),
            'confidence': trial.suggest_float('confidence', 1e-5, 1e-1, log=True),
            'tie_threshold': trial.suggest_float('tie_threshold', 0.01, 0.2),
            'leaf_prediction': trial.suggest_categorical('leaf_prediction', ['MajorityClass', 'NaiveBayes', 'NaiveBayesAdaptive']),
            'nb_threshold': trial.suggest_int('nb_threshold', 0, 50),
            'binary_split': trial.suggest_categorical('binary_split', [True, False]),
            'remove_poor_attrs': trial.suggest_categorical('remove_poor_attrs', [True, False]),
            'disable_prepruning': trial.suggest_categorical('disable_prepruning', [True, False])
        }
        models = get_classification_models(self.schema, selected_models=['HAT'], hat_params=hat_params)
        return self._evaluate_model(models['HoeffdingAdaptiveTree']) 

    def _objective_arf(self, trial):
        arf_params = {
            'ensemble_size': trial.suggest_int('ensemble_size', 10, 150, step=10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0, step=0.1),
            'lambda_param': trial.suggest_float('lambda_param', 1.0, 10.0, step=1.0),
            'disable_weighted_vote': trial.suggest_categorical('disable_weighted_vote', [True, False]),
            'disable_drift_detection': trial.suggest_categorical('disable_drift_detection', [True, False]),
            'disable_background_learner': trial.suggest_categorical('disable_background_learner', [True, False])
        }
        models = get_classification_models(self.schema, selected_models=['ARF'], arf_params=arf_params)
        return self._evaluate_model(models['AdaptiveRandomForest']) 

    def _objective_ht(self, trial):
        ht_params = {
            'grace_period': trial.suggest_int('grace_period', 10, 500, step=10),
            'split_criterion': trial.suggest_categorical('split_criterion', ['InfoGainSplitCriterion', 'GiniSplitCriterion', 'HellingerDistanceCriterion']),
            'confidence': trial.suggest_float('confidence', 1e-5, 1e-1, log=True),
            'tie_threshold': trial.suggest_float('tie_threshold', 0.01, 0.2),
            'leaf_prediction': trial.suggest_categorical('leaf_prediction', ['MajorityClass', 'NaiveBayes', 'NaiveBayesAdaptive']),
            'nb_threshold': trial.suggest_int('nb_threshold', 0, 50),
            'binary_split': trial.suggest_categorical('binary_split', [True, False]),
            'remove_poor_attrs': trial.suggest_categorical('remove_poor_attrs', [True, False]),
            'disable_prepruning': trial.suggest_categorical('disable_prepruning', [True, False])
        }
        models = get_classification_models(self.schema, selected_models=['HT'], ht_params=ht_params)
        return self._evaluate_model(models['HoeffdingTree'])