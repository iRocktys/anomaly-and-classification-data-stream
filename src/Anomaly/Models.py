from capymoa.anomaly import (
    HalfSpaceTrees,
    Autoencoder,
    AdaptiveIsolationForest,
)

def get_anomaly_models(
    schema, 
    selected_models=None, 
    hst_params=None, 
    ae_params=None,  
    aif_params=None,
    run_seed=None
):
    if selected_models is None:
        selected_models = ['HST', 'AE', 'AIF']
    
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