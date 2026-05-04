from capymoa.classifier import (
    LeveragingBagging,
    HoeffdingTree,
    HoeffdingAdaptiveTree,
    AdaptiveRandomForestClassifier,
)

def get_classification_models(schema, selected_models=None, lb_params=None, hat_params=None, arf_params=None, ht_params=None, run_seed=None):
    if selected_models is None:
        selected_models = ['LB', 'HAT', 'ARF', 'HT']
    
    models = {}

    if 'LB' in selected_models:
        default_lb = {
            'schema': schema, 
            'CLI': None, 
            'random_seed': 1 if run_seed is None else run_seed, 
            'base_learner': None,
            'ensemble_size': 100, 
            'minibatch_size': None, 
            'number_of_jobs': None
        }
        if lb_params: default_lb.update(lb_params)
        models["LeveragingBagging"] = LeveragingBagging(**default_lb)

    if 'HAT' in selected_models:
        default_hat = {
            'schema': schema, 
            'random_seed': 0 if run_seed is None else run_seed, 
            'grace_period': 200,
            'split_criterion': 'InfoGainSplitCriterion', 
            'confidence': 0.001,
            'tie_threshold': 0.05, 
            'leaf_prediction': 'NaiveBayesAdaptive',
            'nb_threshold': 0, 
            'numeric_attribute_observer': 'GaussianNumericAttributeClassObserver',
            'binary_split': False, 
            'max_byte_size': 33554432, 
            'memory_estimate_period': 1000000,
            'stop_mem_management': True, 
            'remove_poor_attrs': False, 
            'disable_prepruning': True
        }
        if hat_params: default_hat.update(hat_params)
        models["HoeffdingAdaptiveTree"] = HoeffdingAdaptiveTree(**default_hat)

    if 'ARF' in selected_models:
        default_arf = {
            'schema': schema, 
            'CLI': None, 
            'random_seed': 1 if run_seed is None else run_seed, 
            'base_learner': None,
            'ensemble_size': 100, 
            'max_features': 0.6, 
            'lambda_param': 6.0,
            'minibatch_size': None, 
            'number_of_jobs': 1, 
            'drift_detection_method': None,
            'warning_detection_method': None, 
            'disable_weighted_vote': False,
            'disable_drift_detection': False, 
            'disable_background_learner': False
        }
        if arf_params: default_arf.update(arf_params)
        models["AdaptiveRandomForest"] = AdaptiveRandomForestClassifier(**default_arf)

    if 'HT' in selected_models:
        default_ht = {
            'schema': schema, 
            'random_seed': 1 if run_seed is None else run_seed,
            'grace_period': 200,
            'split_criterion': 'InfoGainSplitCriterion', 
            'confidence': 0.001,
            'tie_threshold': 0.05, 
            'leaf_prediction': 'NaiveBayesAdaptive',
            'nb_threshold': 0, 
            'numeric_attribute_observer': 'GaussianNumericAttributeClassObserver',
            'binary_split': False, 
            'max_byte_size': 33554432, 
            'memory_estimate_period': 1000000,
            'stop_mem_management': True, 
            'remove_poor_attrs': False, 
            'disable_prepruning': True
        }
        if ht_params: default_ht.update(ht_params)
        models["HoeffdingTree"] = HoeffdingTree(**default_ht)

    return models