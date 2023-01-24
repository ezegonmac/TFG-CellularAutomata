from constants import *
from learning.evaluation import evaluate_nn_model_gs, evaluate_nn_model_gsh


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
