from time import time

import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from constants import *
from learning.density_dataset import get_dataset_density_train_test_split
from utils import *


def _evaluate_model_gsh(model, dataset, param_grid, suffix=''):
    split = get_dataset_density_train_test_split(dataset, scaled=True)
    _, _, X_train, _, y_train, _ = split
    
    # Halving Grid Search
    tic = time()
    gsh = HalvingGridSearchCV(
        estimator=model, param_grid=param_grid, factor=2, random_state=SKLEARN_RANDOM_SEED,
        verbose=0, n_jobs=-1, cv=5,
        aggressive_elimination=True
    )
    gsh.fit(X_train, y_train)
    
    gsh_time = time() - tic
    print('HGS time: ', gsh_time)
    best_params = gsh.best_params_
    print('Best params: ', best_params)
    score = gsh.best_score_
    print('Best score: ', score)
    
    model_name = model.__class__.__name__
    
    # Save results
    results = gsh.cv_results_
    df = pd.DataFrame(results)
    suffix = f'_{suffix}' if suffix else ''
    data_evaluation_folder = get_data_evaluation_folder(dataset)
    df.to_csv(f'{data_evaluation_folder}/gsh_{model_name}_results_{dataset}{suffix}.csv')


def _evaluate_model_gs(model, dataset, param_grid, suffix=''):
    split = get_dataset_density_train_test_split(dataset, scaled=True)
    _, _, X_train, _, y_train, _ = split
    
    # Grid Search
    tic = time()
    gs = GridSearchCV(
        estimator=model, param_grid=param_grid, 
        verbose=0, n_jobs=-1, cv=5
    )
    gs.fit(X_train, y_train)
    
    gs_time = time() - tic
    print('GS time: ', gs_time)
    best_params = gs.best_params_
    print('Best params: ', best_params)
    score = gs.best_score_
    print('Best score: ', score)
    
    model_name = model.__class__.__name__
   
    # Save results
    results = gs.cv_results_
    df = pd.DataFrame(results)
    suffix = f'_{suffix}' if suffix else ''
    data_evaluation_folder = get_data_evaluation_folder(dataset)
    df.to_csv(f'{data_evaluation_folder}/gs_{model_name}_results_{dataset}{suffix}.csv')


# KNN - Grid search (GS) and Halving grid search (HGS)

def evaluate_knn_model_gsh(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Halving Search (HGS) 
    to find the best hyperparameters for a NN model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """    
    
    knn_model = KNeighborsRegressor()
    
    _evaluate_model_gsh(knn_model, dataset, param_grid, suffix)


def evaluate_knn_model_gs(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Search (GS) 
    to find the best hyperparameters for a KNN model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """   
    
    knn_model = KNeighborsRegressor()
    
    _evaluate_model_gs(knn_model, dataset, param_grid, suffix)


# DTREE - Grid search (GS) and Halving grid search (HGS)

def evaluate_dtree_model_gsh(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Halving Search (HGS) 
    to find the best hyperparameters for a DTREE model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """    
    
    dtree_model = DecisionTreeRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    _evaluate_model_gsh(dtree_model, dataset, param_grid, suffix)


def evaluate_dtree_model_gs(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Search (GS) 
    to find the best hyperparameters for a KNN model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """   
    
    dtree_model = DecisionTreeRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    _evaluate_model_gs(dtree_model, dataset, param_grid, suffix)


# RF - Grid search (GS) and Halving grid search (HGS)

def evaluate_rf_model_gsh(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Halving Search (HGS) 
    to find the best hyperparameters for a RF model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """    
    
    rf_model = RandomForestRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    _evaluate_model_gsh(rf_model, dataset, param_grid, suffix)


def evaluate_rf_model_gs(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Search (GS) 
    to find the best hyperparameters for a KNN model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """   
    
    rf_model = RandomForestRegressor(random_state=SKLEARN_RANDOM_SEED)
     
    _evaluate_model_gs(rf_model, dataset, param_grid, suffix)


# NN - Grid search (GS) and Halving grid search (HGS)

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
    
    nn_model = MLPRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    _evaluate_model_gsh(nn_model, dataset, param_grid, suffix)


def evaluate_nn_model_gs(dataset, param_grid, suffix=''):
    """
    Given a dataset and a search space,
    it performs a Grid Search (GS) 
    to find the best hyperparameters for a KNN model.

    Args:
        dataset (str): name of the dataset
        param_grid (dict): search space with all the hyperparameters to be tested
        suffix (str, optional): suffix to be added to the name of the file where the results are saved
    """   
    
    nn_model = MLPRegressor(random_state=SKLEARN_RANDOM_SEED)
    
    _evaluate_model_gs(nn_model, dataset, param_grid, suffix)
