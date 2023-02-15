from constants import *
from learning.evaluation.searchs import (
    evaluate_knn_model_gs, evaluate_knn_model_gsh,
    evaluate_dtree_model_gs, evaluate_dtree_model_gsh,
    evaluate_rf_model_gs, evaluate_rf_model_gsh,
    evaluate_nn_model_gs, evaluate_nn_model_gsh)

def evaluate_nn_model_ds9_1():
    dataset = DATASET9_DENSITY
    
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


def evaluate_nn_model_ds9_2():
    dataset = DATASET9_DENSITY
    
    param_grid = {
        'hidden_layer_sizes': [(18, 18, 18), (18, 36, 18), (36, 36, 36), (36, 18, 36), (36, 36, 18), (18, 36, 36), (36, 18, 18), (18, 18, 36), (20, 40, 20), (20, 40, 40), (50, 50, 50)],
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
        'early_stopping': [True],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='2')


def evaluate_nn_model_ds9_3():
    dataset = DATASET9_DENSITY
    
    param_grid = {
        'hidden_layer_sizes': [(50, 50, 50), (64, 64, 64), (80, 80, 80), (70, 70, 70), (60, 60, 60), (90, 90, 90), (50, 50, 50, 50), (64, 64, 64, 64), (80, 80, 80, 80), (70, 70, 70, 70), (60, 60, 60, 60), (90, 90, 90, 90)],
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
        'early_stopping': [True],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='3')


def evaluate_nn_model_ds9_4():
    dataset = DATASET9_DENSITY
    
    param_grid = {
        'hidden_layer_sizes': [(60, 60, 60, 60), (90, 90, 90, 90), (60, 60, 60, 60, 60), (50, 50, 50, 50, 50), (70, 70, 70), (70, 70, 70, 70), (70, 70, 70, 70, 70), (64, 64, 64, 64)],
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
        'early_stopping': [True],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='4')


def evaluate_knn_model_ds9_1():
    dataset = DATASET9_DENSITY
    
    param_grid = {
        'n_neighbors': range(1, 20, 2),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': range(1, 50, 5),
        'p': [1, 2],
    }
    
    evaluate_knn_model_gs(dataset, param_grid, suffix='1')


def evaluate_dtree_model_ds9_1():
    dataset = DATASET9_DENSITY
    
    param_grid = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    
    evaluate_dtree_model_gs(dataset, param_grid, suffix='1')


def evaluate_rf_model_ds9_1():
    dataset = DATASET9_DENSITY
    
    param_grid = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error'],
        'n_estimators': range(1, 20, 2),
        'max_depth': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    
    evaluate_rf_model_gs(dataset, param_grid, suffix='1')