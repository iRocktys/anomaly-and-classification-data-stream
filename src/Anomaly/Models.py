from capymoa.anomaly import (
    HalfSpaceTrees,
    OnlineIsolationForest,
    Autoencoder,
    StreamingIsolationForest,
    RobustRandomCutForest,
    AdaptiveIsolationForest,
)

def get_anomaly_models(
    schema, 
    selected_models=None, 
    hst_params=None, 
    oif_params=None, 
    ae_params=None,  
    rrcf_params=None, 
    aif_params=None,
    run_seed=None
):
    if selected_models is None:
        selected_models = ['HST', 'OIF', 'AE', 'RRCF', 'AIF']
    
    models = {}

    if 'HST' in selected_models:
        default_hst = {
            'schema': schema,
            'CLI': None,
            'random_seed': 1 if run_seed is None else run_seed,
            'window_size': 250,
            'number_of_trees': 25,
            'max_depth': 15,
            'anomaly_threshold': 0.50,
            'size_limit': 0.10
        }
        if hst_params: default_hst.update(hst_params)
        models["HalfSpaceTrees"] = HalfSpaceTrees(**default_hst)

    if 'OIF' in selected_models:
        default_oif = {
            'schema': schema,
            'random_seed': 1 if run_seed is None else run_seed,
            'num_trees': 100,
            'max_leaf_samples': 16,
            'growth_criterion': 'adaptive',
            'subsample': 0.9,
            'window_size': 2048,
            'branching_factor': 4,
            'split': 'axisparallel',
            'n_jobs': 1
        }
        if oif_params: default_oif.update(oif_params)
        models["OnlineIsolationForest"] = OnlineIsolationForest(**default_oif)

    if 'AE' in selected_models:
        default_ae = {
            'schema': schema,
            'hidden_layer': 2,
            'learning_rate': 0.5,
            'threshold': 0.6,
            'random_seed': 1 if run_seed is None else run_seed
        }
        if ae_params: default_ae.update(ae_params)
        models["Autoencoder"] = Autoencoder(**default_ae)

    if 'RRCF' in selected_models:
        default_rrcf = {
            'schema': schema, 
            'tree_size': 1000,
            'n_trees': 100,
            'random_state': 42 if run_seed is None else run_seed
        }
        if rrcf_params: default_rrcf.update(rrcf_params)
        models["RobustRandomCutForest"] = RobustRandomCutForest(**default_rrcf)

    if 'AIF' in selected_models:
        default_aif = {
            'schema': schema, 
            'window_size': 256,
            'n_trees': 100,
            'height': None,
            'seed': None if run_seed is None else run_seed,
            'm_trees': 10,
            'weights': 0.5
        }
        if aif_params: default_aif.update(aif_params)
        models["AdaptiveIsolationForest"] = AdaptiveIsolationForest(**default_aif)

    return models