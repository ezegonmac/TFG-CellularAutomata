from constants import *
from learning.evaluation.searchs import (
    evaluate_knn_model_gs, evaluate_knn_model_gsh,
    evaluate_dtree_model_gs, evaluate_dtree_model_gsh,
    evaluate_rf_model_gs, evaluate_rf_model_gsh,
    evaluate_nn_model_gs, evaluate_nn_model_gsh)


def evaluate_nn_model_ds12_1():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(36, 36, 36), (100, 100, 100), (9, 18, 9), (18, 18, 18), (18, 9, 18), (10, 18, 9)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='1')


def evaluate_nn_model_ds12_2():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
    }
    
    evaluate_nn_model_gsh(dataset, param_grid, suffix='2')


def evaluate_nn_model_ds12_3():
    dataset = DATASET12_DENSITY
    param_grid = {
        'hidden_layer_sizes': [(100, 100, 100), (100, 100, 100, 100), (50, 100, 50), (50, 50, 50), (50, 50, 50, 50)],
        'activation': ['tanh'],
        'solver': ['lbfgs'],
        'alpha': [0.05],
        'learning_rate': ['invscaling'],
        'max_iter': [5000],
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
    
    # TODO: not call this function here ?
    generate_scores_file(X, y, X_test, y_test, y_pred, model, dataset=dataset, model_name=model_name)
    print_scores(dataset, model_name)
    

def evaluate_knn_model_ds12_1():
    dataset = DATASET12_DENSITY
    
    param_grid = {
        'n_neighbors': range(1, 20, 2),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': range(1, 50, 5),
        'p': [1, 2],
    }
    
    evaluate_knn_model_gs(dataset, param_grid, suffix='1')


def evaluate_dtree_model_ds12_1():
    dataset = DATASET12_DENSITY
    
    param_grid = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    
    evaluate_dtree_model_gsh(dataset, param_grid, suffix='1')


def evaluate_rf_model_ds12_1():
    dataset = DATASET12_DENSITY
    
    param_grid = {
        'criterion': ['poisson', 'friedman_mse', 'squared_error'],
        'n_estimators': range(1, 20, 2),
        'max_depth': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2),
        'max_features': ['auto', 'sqrt', 'log2', None],
    }
    
    evaluate_rf_model_gsh(dataset, param_grid, suffix='1')

