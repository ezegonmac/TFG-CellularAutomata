from time import time

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor

from constants import *
from learning.BTST_density import get_dataset_density_X_y_split

EVALUATION_FILE = f'{DATA_LEARNING_FOLDER}/evaluations.csv'


def evaluate_nn_model_ds3_1():
    dataset = DATASET3_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100), (9, 18, 9), (9, 9, 9), (18, 18, 18), (18, 9, 18)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.05],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [3000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='1')


def evaluate_nn_model_gsh(dataset, param_grid, suffix=''):
    
    X, y = get_dataset_density_X_y_split(dataset)
    
    nn_model = MLPRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    # Halving Grid Search
    tic = time()
    gsh = HalvingGridSearchCV(
        estimator=nn_model, param_grid=param_grid, factor=2, random_state=SKLEARN_RANDOM_SEED,
        verbose=0, n_jobs=-1, cv=5,
        aggressive_elimination=True
    )
    gsh.fit(X, y)
    
    gsh_time = time() - tic
    print('HGS time: ', gsh_time)
    best_params = gsh.best_params_
    print('Best params: ', best_params)
    score = gsh.best_score_
    print('Best score: ', score)
    
    results = gsh.cv_results_
    df = pd.DataFrame(results)
    suffix = f'_{suffix}' if suffix else ''
    df.to_csv(f'{DATA_LEARNING_FOLDER}/{dataset}/gsh_results_{dataset}{suffix}.csv')



def evaluate_nn_model_gs(dataset, param_grid):
    
    X, y = get_dataset_density_X_y_split(dataset)
    
    nn_model = MLPRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    # Grid Search
    tic = time()
    gs = GridSearchCV(
        estimator=nn_model, param_grid=param_grid, random_state=SKLEARN_RANDOM_SEED,
        verbose=0, n_jobs=-1, cv=5
    )
    gs.fit(X, y)
    
    gs_time = time() - tic
    print('GS time: ', gs_time)
    best_params = gs.best_params_
    print('Best params: ', best_params)
    score = gs.best_score_
    print('Best score: ', score)
