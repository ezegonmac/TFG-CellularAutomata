import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from constants import *
from datasets.density_datasets_BISI import *
from learning.density_dataset import *
from learning.test.files import generate_model_and_scores_files
from learning.test.plots import (generate_score_evolution_comparison_plots,
                                 generate_score_evolution_plots,
                                 generate_scores_model_comparison_plot,
                                 generate_score_model_comparison_plots)
from utils import *

# Train and test models

def train_and_test_models(dataset, model_variation='vector', hyperparameters_knn={}, hyperparameters_dtree={}, hyperparameters_rf= {}, hyperparameters_nn={}, num_executions=10, num_individuals=500, save_models=False):
    knn_model = KNeighborsRegressor(**hyperparameters_knn)
    dtree_model = DecisionTreeRegressor(random_state=SKLEARN_RANDOM_SEED, **hyperparameters_dtree)
    rf_model = RandomForestRegressor(random_state=SKLEARN_RANDOM_SEED, **hyperparameters_rf)
    nn_model = MLPRegressor(
        random_state=SKLEARN_RANDOM_SEED,
        # verbose=True
        **hyperparameters_nn,
        )
    
    generate_model_and_scores_files(knn_model, dataset, 'KNN', model_variation, save_model=save_models, num_individuals=num_individuals, num_executions=num_executions)
    generate_model_and_scores_files(dtree_model, dataset, 'DecisionTree', model_variation, save_model=save_models, num_individuals=num_individuals, num_executions=num_executions)
    generate_model_and_scores_files(rf_model, dataset, 'RandomForest', model_variation, save_model=save_models, num_individuals=num_individuals, num_executions=num_executions)
    generate_model_and_scores_files(nn_model, dataset, 'NeuralNetwork', model_variation, save_model=save_models, num_individuals=num_individuals, num_executions=num_executions)

# Generate dataset

def generate_dataset(dataset, num_executions=10, num_individuals=500):
    print(f"# Generating dataset with {num_executions} executions and {num_individuals} individuals")
    dataset_individuals = num_executions * num_individuals
    generate_dataset = globals()[f'generate_{dataset}']
    generate_dataset(dataset_individuals)

# ---------  #
#  DATASETS  #
# ---------  #

# DATASET 11 #

def train_and_test_models_ds11(num_executions, num_individuals, save_models=False):
    dataset = DATASET11_DENSITY
    
    print('---------------------------------')
    print('Train and test: ' + dataset)
    print('---------------------------------')
    
    # Hyperparameters
    
    hp_knn = {
        'algorithm': 'auto', 
        'leaf_size': 1, 
        'n_neighbors': 7, 
        'p': 1, 
        'weights': 'distance'
    }
    
    hp_dtree = {
        'criterion': 'squared_error', 
        'max_depth': 15, 
        'max_features': 'sqrt', 
        'min_samples_leaf': 9, 
        'min_samples_split': 4, 
        'splitter': 'best'
    }
    
    hp_rf = {
        'criterion': 'squared_error', 
        'max_depth': 11, 
        'max_features': 'auto', 
        'min_samples_leaf': 1, 
        'min_samples_split': 2, 
        'n_estimators': 15
    }
    
    hp_nn = {
        'hidden_layer_sizes': (36, 36, 36),
        'max_iter': 5000,
        'solver': 'lbfgs',
        'activation': 'logistic',
        'alpha': 0.05,
        'learning_rate': 'invscaling',
        'early_stopping': True,
    }
    
    start_time = time.time()
    
    generate_dataset(
        dataset,
        num_executions=num_executions, 
        num_individuals=num_individuals
        )
    
    train_and_test_models(
        dataset,
        hyperparameters_knn=hp_knn,
        hyperparameters_dtree=hp_dtree,
        hyperparameters_rf=hp_rf,
        hyperparameters_nn=hp_nn,
        model_variation='vector',
        num_individuals=num_individuals,
        save_models=save_models,
        num_executions=num_executions,
        )
    
    # time in minutes
    elapsed_time = (time.time() - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes\n")

# DATASET 12 #

def train_and_test_models_ds12(num_executions, num_individuals, save_models=False):
    dataset = DATASET12_DENSITY
    
    print('---------------------------------')
    print('Train and test: ' + dataset)
    print('---------------------------------')
    
    # Hyperparameters
    
    hp_knn = {
        'algorithm': 'brute', 
        'leaf_size': 1, 
        'n_neighbors': 7, 
        'p': 2, 
        'weights': 'distance'
    }
    
    hp_dtree = {
        'criterion': 'squared_error', 
        'max_depth': 15, 
        'max_features': None, 
        'min_samples_leaf': 1, 
        'min_samples_split': 10, 
        'splitter': 'random'
    }
    
    hp_rf = {
        'criterion': 'friedman_mse', 
        'max_depth': 11, 
        'max_features': None, 
        'min_samples_leaf': 5, 
        'min_samples_split': 12, 
        'n_estimators': 1
    }
    
    hp_nn = {
        'hidden_layer_sizes': (36, 36, 36),
        'max_iter': 5000,
        'solver': 'lbfgs',
        'activation': 'logistic',
        'alpha': 0.05,
        'learning_rate': 'invscaling',
        'early_stopping': True,
    }
    
    start_time = time.time()
    
    generate_dataset(
        dataset,
        num_executions=num_executions, 
        num_individuals=num_individuals
        )
    
    train_and_test_models(
        dataset,
        hyperparameters_knn=hp_knn,
        hyperparameters_dtree=hp_dtree,
        hyperparameters_rf=hp_rf,
        hyperparameters_nn=hp_nn,
        model_variation='vector',
        num_individuals=num_individuals,
        save_models=save_models,
        num_executions=num_executions,
        )
    
    # time in minutes
    elapsed_time = (time.time() - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes\n")
    
# DATASET 13 #

def train_and_test_models_ds13(num_executions, num_individuals, save_models=False):
    dataset = DATASET13_DENSITY
    
    print('---------------------------------')
    print('Train and test: ' + dataset)
    print('---------------------------------')
    
    # Hyperparameters
    
    hp_knn = {
        'algorithm': 'brute', 
        'leaf_size': 1, 
        'n_neighbors': 9, 
        'p': 1, 
        'weights': 'distance'
    }

    hp_dtree = {
        'criterion': 'squared_error', 
        'max_depth': 13, 
        'max_features': 'auto', 
        'min_samples_leaf': 1, 
        'min_samples_split': 6, 
        'splitter': 'random'
    }
    
    hp_rf = {
        'criterion': 'squared_error', 
        'max_depth': 13, 
        'max_features': 'sqrt', 
        'min_samples_leaf': 1, 
        'min_samples_split': 6, 
        'n_estimators': 5
    }
    
    hp_nn = {
        'hidden_layer_sizes': (36, 36, 36),
        'max_iter': 5000,
        'solver': 'lbfgs',
        'activation': 'logistic',
        'alpha': 0.05,
        'learning_rate': 'invscaling',
        'early_stopping': True,
    }
    
    start_time = time.time()
    
    generate_dataset(
        dataset,
        num_executions=num_executions, 
        num_individuals=num_individuals
        )
    
    train_and_test_models(
        dataset,
        hyperparameters_knn=hp_knn,
        hyperparameters_dtree=hp_dtree,
        hyperparameters_rf=hp_rf,
        hyperparameters_nn=hp_nn,
        model_variation='vector',
        num_individuals=num_individuals,
        save_models=save_models,
        num_executions=num_executions,
        )
    
    # time in minutes
    elapsed_time = (time.time() - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} minutes\n")

# --------  #
#   PLOTS   # 
# --------  #

def generate_models_score_plots_ds11(num_individuals):
    dataset = DATASET11_DENSITY
    
    # score model comparison plots
    # generate_score_model_comparison_plots(dataset, num_individuals=num_individuals)
    # score individual evolution plots
    # generate_score_evolution_plots(dataset, num_individuals=num_individuals)
    # score evolution comparison plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)


def generate_models_score_plots_ds12(num_individuals):
    dataset = DATASET12_DENSITY
    
    # score model comparison plots
    generate_score_model_comparison_plots(dataset, num_individuals=num_individuals)
    # score individual evolution plots
    # generate_score_evolution_plots(dataset, num_individuals=num_individuals)
    # score evolution comparison plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)


def generate_models_score_plots_ds13(num_individuals):
    dataset = DATASET13_DENSITY
    
    # score model comparison plots
    generate_score_model_comparison_plots(dataset, num_individuals=num_individuals)
    # score individual evolution plots
    # generate_score_evolution_plots(dataset, num_individuals=num_individuals)
    # score evolution comparison plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)
