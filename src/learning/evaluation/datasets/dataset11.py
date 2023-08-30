from constants import *
from learning.evaluation.searchs import (
    evaluate_knn_model_gs, evaluate_knn_model_gsh,
    evaluate_dtree_model_gs, evaluate_dtree_model_gsh,
    evaluate_rf_model_gs, evaluate_rf_model_gsh,
    evaluate_nn_model_gs, evaluate_nn_model_gsh)

def evaluate_nn_model_ds11_1():
    dataset = DATASET11_DENSITY
    
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100), (9, 18, 9), (9, 9, 9), (18, 18, 18), (18, 9, 18)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.05],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [5000],
        'early_stopping': [True],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='1')


def evaluate_nn_model_ds11_2():
    dataset = DATASET11_DENSITY
    
    param_grid = {
        'hidden_layer_sizes': [(9, 18, 9), (18, 36, 18), (36, 36, 36), (36, 18, 36), (36, 36, 18), (18, 36, 36), (36, 18, 18), (18, 18, 36), (20, 40, 20), (20, 40, 40), (50, 50, 50)],
        'activation': ['logistic'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
        'early_stopping': [True],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='2')


def evaluate_nn_model_ds11_3():
    dataset = DATASET11_DENSITY
    
    param_grid = {
        'hidden_layer_sizes': [(36, 36, 36), (36, 18, 18), (18, 18, 36), (20, 40, 20), (20, 40, 40), (50, 50, 50), (60, 60, 60, 60), (60, 60, 60, 30), (60, 60, 30, 60), (60, 30, 60, 60), (30, 60, 60, 60), (60, 60, 30, 30), (60, 30, 30, 60), (30, 60, 60, 30), (30, 30, 60, 60), (30, 30, 30, 60), (30, 30, 60, 30), (30, 30, 30, 30)],
        'activation': ['logistic'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
        'early_stopping': [True],
    }
    
    evaluate_nn_model_gs(dataset, param_grid, suffix='3')


def evaluate_knn_model_ds11_1():
    dataset = DATASET11_DENSITY
    
    param_grid = {
        'n_neighbors': range(1, 20, 2),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': range(1, 50, 5),
        'p': [1, 2],
    }
    
    evaluate_knn_model_gs(dataset, param_grid, suffix='1')


def evaluate_dtree_model_ds11_1():
    dataset = DATASET11_DENSITY
    
    param_grid = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    
    evaluate_dtree_model_gs(dataset, param_grid, suffix='1')


def evaluate_rf_model_ds11_1():
    dataset = DATASET11_DENSITY
    
    param_grid = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error'],
        'n_estimators': range(1, 20, 2),
        'max_depth': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    
    evaluate_rf_model_gsh(dataset, param_grid, suffix='1')
