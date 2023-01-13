from time import time

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor

from constants import *
from learning.BTST_density import get_dataset_density_X_y_split

EVALUATION_FILE = f'{DATA_LEARNING_FOLDER}/evaluations.csv'


def evaluate_nn_model(dataset):
    
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100), (9, 18, 9), (9, 9, 9), (18, 18, 18), (18, 9, 18)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [500, 1000, 2000],
    }
    
    X, y = get_dataset_density_X_y_split(dataset)
    
    nn_model = MLPRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    # Halving Grid Search
    tic = time()
    gsh = HalvingGridSearchCV(
        estimator=nn_model, param_grid=param_grid, factor=2, random_state=SKLEARN_RANDOM_SEED,
        verbose=1, n_jobs=-1, cv=5, scoring='neg_mean_absolute_error',
        aggressive_elimination=True
    )
    gsh.fit(X, y)
    
    gsh_time = time() - tic
    print('HGS time: ', gsh_time)
    best_params = gsh.best_params_
    print('Best params: ', best_params)
    
    # # Grid Search
    # tic = time()
    # gs = GridSearchCV(
    #     estimator=nn_model, param_grid=param_grid, random_state=SKLEARN_RANDOM_SEED,
    #     verbose=1, n_jobs=-1, cv=5, scoring='neg_mean_absolute_error'
    # )
    # gs.fit(X, y)
    
    # gs_time = time() - tic
    # print('GS time: ', gs_time)
    # best_params = ghs.best_params_
    # print('Best params: ', best_params)
