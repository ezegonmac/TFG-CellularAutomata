from constants import *
from learning.evaluation.searchs import (
    evaluate_knn_model_gs, evaluate_knn_model_gsh,
    evaluate_dtree_model_gs, evaluate_dtree_model_gsh,
    evaluate_rf_model_gs, evaluate_rf_model_gsh,
    evaluate_nn_model_gs, evaluate_nn_model_gsh)


def evaluate_nn_model_ds14_1():
    dataset = DATASET14_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(36, 36, 36), (100, 100, 100), (9, 18, 9), (18, 18, 18), (18, 9, 18), (10, 18, 9)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='1')


def evaluate_nn_model_ds14_2():
    dataset = DATASET14_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='2')


def evaluate_nn_model_ds14_3():
    dataset = DATASET14_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100), (100, 100, 100, 100), (50, 100, 50), (50, 50, 50), (50, 50, 50, 50)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='3')
    

def evaluate_knn_model_ds14_1():
    dataset = DATASET14_DENSITY
    
    param_grid = {
        'n_neighbors': range(1, 20, 2),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': range(1, 50, 5),
        'p': [1, 2],
    }
    
    evaluate_knn_model_gs(dataset, param_grid, suffix='1')


def evaluate_dtree_model_ds14_1():
    dataset = DATASET14_DENSITY
    
    param_grid = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    
    evaluate_dtree_model_gsh(dataset, param_grid, suffix='1')


def evaluate_rf_model_ds14_1():
    dataset = DATASET14_DENSITY
    
    param_grid = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error'],
        'n_estimators': range(1, 20, 2),
        'max_depth': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    
    evaluate_rf_model_gsh(dataset, param_grid, suffix='1')
