from time import time

import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor

from constants import *
from learning.density_dataset import get_dataset_density_train_test_split
from utils import *

# Grid search (GS) and Halving grid search (HGS)

def evaluate_nn_model_gsh(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Halving Search (HGS) 
    to find the best hyperparameters for a NN model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """    
    
    split = get_dataset_density_train_test_split(dataset, scaled=True)
    _, _, X_train, _, y_train, _ = split
    
    # Model
    nn_model = MLPRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    # Halving Grid Search
    tic = time()
    gsh = HalvingGridSearchCV(
        estimator=nn_model, param_grid=param_grid, factor=2, random_state=SKLEARN_RANDOM_SEED,
        verbose=1, n_jobs=-1, cv=5,
        aggressive_elimination=True
    )
    gsh.fit(X_train, y_train)
    
    gsh_time = time() - tic
    print('HGS time: ', gsh_time)
    best_params = gsh.best_params_
    print('Best params: ', best_params)
    score = gsh.best_score_
    print('Best score: ', score)
    
    # Save results
    results = gsh.cv_results_
    df = pd.DataFrame(results)
    suffix = f'_{suffix}' if suffix else ''
    data_learning_folder = get_data_learning_folder(dataset)
    df.to_csv(f'{data_learning_folder}/{dataset}/gsh_results_{dataset}{suffix}.csv')


def evaluate_nn_model_gs(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Search (GS) 
    to find the best hyperparameters for a NN model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """   
    
    split = get_dataset_density_train_test_split(dataset, scaled=True)
    _, _, X_train, _, y_train, _ = split
    
    # Model
    nn_model = MLPRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    # Grid Search
    tic = time()
    gs = GridSearchCV(
        estimator=nn_model, param_grid=param_grid, random_state=SKLEARN_RANDOM_SEED,
        verbose=1, n_jobs=-1, cv=5
    )
    gs.fit(X_train, y_train)
    
    gs_time = time() - tic
    print('GS time: ', gs_time)
    best_params = gs.best_params_
    print('Best params: ', best_params)
    score = gs.best_score_
    print('Best score: ', score)

    # Save results
    results = gs.cv_results_
    df = pd.DataFrame(results)
    suffix = f'_{suffix}' if suffix else ''
    data_learning_folder = get_data_learning_folder(dataset)
    df.to_csv(f'{data_learning_folder}/{dataset}/gsh_results_{dataset}{suffix}.csv')
