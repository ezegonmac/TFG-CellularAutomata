from ray.util.joblib import register_ray
from joblib import parallel_backend

def start_parallelization():
    register_ray()

def fit_model_parallel(model, X_train, y_train):
    with parallel_backend('ray'):
        model.fit(X_train, y_train)
        
        return model
