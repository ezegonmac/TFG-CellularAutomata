from time import time

import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from constants import *
from learning.BTST_density import *


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


def evaluate_nn_model_ds12_1():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100), (9, 18, 9), (18, 18, 18), (18, 9, 18), (10, 18, 9)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [10000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='1')


def evaluate_nn_model_ds12_2():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': ['invscaling'],
        'max_iter': [10000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='2')


def evaluate_nn_model_ds12_3():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100), (100, 100, 100, 100), (50, 100, 50), (50, 50, 50), (50, 50, 50, 50)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.3],
        'learning_rate': ['invscaling'],
        'max_iter': [10000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='3')


def evaluate_nn_model_ds12_4():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(50, 50, 50, 50), (50, 100, 100, 50), (25, 50, 50, 25), (50, 50, 50, 50, 50), (25, 50, 50, 50, 25)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.3],
        'learning_rate': ['invscaling'],
        'max_iter': [10000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='4')


def evaluate_nn_model_ds12_5():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(25, 50, 50, 50, 25), (25, 50, 50, 50, 25, 25), (25, 50, 50, 50, 50, 25), (25, 50, 50, 50, 50, 25, 25)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.3],
        'learning_rate': ['invscaling'],
        'max_iter': [10000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='5')


def evaluate_nn_model_ds12_6():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(25, 50, 50, 25, 25), (25, 50, 50, 25, 25, 25), (25, 50, 80, 50, 25), (20, 50, 50, 40, 20), (20, 50, 40, 20)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.3],
        'learning_rate': ['invscaling'],
        'max_iter': [10000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='6')


def evaluate_nn_model_ds12_7():
    # not parallel
    dataset = DATASET12_DENSITY
    model_name = 'NeuralNetwork'
    model = MLPRegressor(
        # parameters
        hidden_layer_sizes=(50, 50, 50, 50),
        alpha=0.3,  # not important
        learning_rate='invscaling',  # medium importance
        max_iter=10000,
        solver='lbfgs',  # important
        activation='tanh',  # not important
        
        random_state=SKLEARN_RANDOM_SEED,
        # batch_size=300,
        early_stopping=True,
        verbose=False,
        )
    
    print('Neural Network')
    print('---------')
    split = get_dataset_density_train_test_split(dataset, scaled=True)
    X, y, X_train, X_test, y_train, y_test = split
    
    tic = time()
    model.fit(X_train, y_train)
    fit_time = time() - tic
    print(f'fit time: {fit_time}')
    
    # model
    generate_model_file(dataset, model, model_name)
    
    # scores
    tic = time()
    y_pred = model.predict(X_test)
    test_time = time() - tic
    print(f'test time: {test_time}')
    
    generate_scores_file(X, y, X_test, y_test, y_pred, model, dataset=dataset, model_name=model_name)
    print_evaluation(dataset, model_name)


# grid search and halving grid search

def evaluate_nn_model_gsh(dataset, param_grid, suffix=''):
    
    split = get_dataset_density_train_test_split(dataset, scaled=True)
    X, y, X_train, X_test, y_train, y_test = split
    
    xscaler = StandardScaler().fit(X_train)
    X_train = xscaler.transform(X_train)
    yscaler = StandardScaler().fit(y_train)
    y_train = yscaler.transform(y_train)
    
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
    
    results = gsh.cv_results_
    df = pd.DataFrame(results)
    
    suffix = f'_{suffix}' if suffix else ''
    data_learning_folder = get_data_learning_folder(dataset)
    df.to_csv(f'{data_learning_folder}/{dataset}/gsh_results_{dataset}{suffix}.csv')


def evaluate_nn_model_gs(dataset, param_grid, suffix=''):
    
    split = get_dataset_density_train_test_split(dataset, scaled=True)
    X, y, X_train, X_test, y_train, y_test = split
    
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

    results = gs.cv_results_
    df = pd.DataFrame(results)
    suffix = f'_{suffix}' if suffix else ''
    data_learning_folder = get_data_learning_folder(dataset)
    df.to_csv(f'{data_learning_folder}/{dataset}/gsh_results_{dataset}{suffix}.csv')
