from constants import *
from learning.evaluation.searchs import evaluate_nn_model_gs, evaluate_nn_model_gsh


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


def evaluate_nn_model_ds3_2():
    dataset = DATASET3_DENSITY
    
    param_grid = {
        'hidden_layer_sizes': [(9, 18, 9)],
        'activation': ['tanh'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.05, 0.001, 0.0001],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [5000],
        'early_stopping': [True],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='2')


def evaluate_nn_model_ds3_3():
    dataset = DATASET3_DENSITY
    
    param_grid = {
        'hidden_layer_sizes': [(9, 18, 9)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.02, 0.05, 0.1, 1.0, 2.0],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
        'early_stopping': [True],
    }
    
    evaluate_nn_model_gs(dataset, param_grid, suffix='3')
