import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from constants import *
from learning.density_dataset import *
from learning.models.files import *
from learning.test import *
from learning.test.plots import *
from datasets.density_datasets_BTST import *
from learning.test.files import generate_model_and_scores_files
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

# DATASET 8 #

def train_and_test_models_ds8(num_executions, num_individuals, save_models=False):
    dataset = DATASET8_DENSITY
    
    print('---------------------------------')
    print('Train and test: ' + dataset)
    print('---------------------------------')
    
    # Hyperparameters
    
    hp_knn = {}
    
    hp_dtree = {}
    
    hp_rf = {}
    
    hp_nn = {
        'hidden_layer_sizes': (3),
        'solver': 'lbfgs',
        'activation': 'tanh',
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


def generate_models_score_plots_ds8(num_individuals):
    dataset = DATASET8_DENSITY
    
    # FIGURES
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='RMSE', num_individuals=num_individuals)
    generate_scores_model_comparison_plot(dataset, metric='R2', num_individuals=num_individuals)

    # score evolution plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)


# DATASET 3 #

def train_and_test_models_ds3(num_executions=10, num_individuals=500, save_models=False):
    dataset = DATASET3_DENSITY
    
    print('---------------------------------')
    print('Train and test: ' + dataset)
    print('---------------------------------')
    
    # Hyperparameters
    
    hp_knn = {
        'n_neighbors': 3,
    }
    
    hp_dtree = {}
    
    hp_rf = {}
    
    hp_nn = {
        'early_stopping': True,
        'hidden_layer_sizes': (9, 18, 9),
        'learning_rate': 'invscaling',
        'max_iter': 5000,
        'solver': 'lbfgs',
        'activation': 'tanh',
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
    
    
def generate_models_score_plots_ds3(num_individuals):
    dataset = DATASET3_DENSITY
    
    # FIGURES
    # # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='RMSE', num_individuals=num_individuals)
    generate_scores_model_comparison_plot(dataset, metric='R2', num_individuals=num_individuals)
    generate_scores_model_comparison_plot(dataset, metric='R2', num_individuals=num_individuals, 
                                          y_min=0.92, y_max=1.0, suffix='scaled')
    
    # # score model comparison individuals plots
    # generate_scores_model_individuals_comparison_plot(dataset, metric='RMSE mean')
    # generate_scores_model_individuals_comparison_plot(dataset, metric='RMSE mean', y_min=0, y_max=0.3, suffix='scaled')
    # generate_scores_model_individuals_comparison_plot(dataset, metric='R2 mean')
    # generate_scores_model_individuals_comparison_plot(dataset, metric='R2 mean', y_min=0.9, y_max=1.0, suffix='scaled')
    
    # # score evolution plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)

    generate_score_evolution_comparison_plot(dataset, metric="R2", y_min=0.85, y_max=1, suffix='scaled', num_individuals=num_individuals)


# DATASET 9 #

def train_and_test_models_ds9(num_executions=10, num_individuals=500, save_models=False):
    dataset = DATASET9_DENSITY
    
    print('---------------------------------')
    print('Train and test: ' + dataset)
    print('---------------------------------')
    
    # Hyperparameters
    hp_knn = {
        'algorithm': 'ball_tree', 
        'leaf_size': 6, 
        'n_neighbors': 11, 
        'p': 2, 
        'weights': 'distance'
    }
    
    hp_dtree = {
        'criterion': 'friedman_mse', 
        'max_depth': 11, 
        'max_features': 'auto', 
        'min_samples_leaf': 3, 
        'min_samples_split': 10, 
        'splitter': 'random'
    }
    
    hp_rf = {
        'criterion': 'squared_error', 
        'max_depth': 9, 
        'max_features': 'auto', 
        'min_samples_leaf': 1, 
        'min_samples_split': 4,
        'n_estimators': 7
        }
    
    hp_nn = {
        'early_stopping': True,
        'hidden_layer_sizes': (60, 60, 60, 60),
        'learning_rate': 'invscaling',
        'max_iter': 5000,
        'solver': 'lbfgs',
        'activation': 'tanh',
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


def generate_models_score_plots_ds9(num_individuals):
    dataset = DATASET9_DENSITY
    
    # # FIGURES
    # # # score model comparison plots
    # generate_scores_model_comparison_plot(dataset, metric='RMSE', num_individuals=num_individuals)
    # generate_scores_model_comparison_plot(dataset, metric='R2', num_individuals=num_individuals)

    # # # score model comparison individuals plots
    # generate_scores_model_individuals_comparison_plot(dataset, metric='RMSE')
    # generate_scores_model_individuals_comparison_plot(dataset, metric='RMSE', y_min=0, y_max=0.6, suffix='scaled')
    # generate_scores_model_individuals_comparison_plot(dataset, metric='R2')
    # generate_scores_model_individuals_comparison_plot(dataset, metric='R2', y_min=0.6, y_max=1.0, suffix='scaled')
    
    # # # score evolution plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)

    # generate_score_evolution_comparison_plot(dataset, metric="R2", y_min=0.85, y_max=1, suffix='scaled')


def train_and_test_models_ds10(num_executions=10, num_individuals=500, save_models=False):
    dataset = DATASET10_DENSITY
    
    print('---------------------------------')
    print('Train and test: ' + dataset)
    print('---------------------------------')
    
    # Hyperparameters
    hp_knn = {
        'algorithm': 'auto', 
        'leaf_size': 1, 
        'n_neighbors': 5, 
        'p': 1, 
        'weights': 'distance'
    }
    
    hp_dtree = {
        'criterion': 'friedman_mse', 
        'max_depth': 13, 
        'max_features': 'auto', 
        'min_samples_leaf': 1, 
        'min_samples_split': 2, 
        'splitter': 'random'
    }
    
    hp_rf = {
        'criterion': 'friedman_mse', 
        'max_depth': 17, 
        'max_features': 'sqrt', 
        'min_samples_leaf': 1, 
        'min_samples_split': 2,
        'n_estimators': 19
        }
    
    hp_nn = {
        'early_stopping': True,
        'hidden_layer_sizes': (36, 18, 18),
        'learning_rate': 'invscaling',
        'max_iter': 5000,
        'alpha': 0.05,
        'solver': 'lbfgs',
        'activation': 'tanh',
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


def generate_models_score_plots_ds10(num_individuals):
    dataset = DATASET10_DENSITY
    
    # FIGURES
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='RMSE', num_individuals=num_individuals)
    generate_scores_model_comparison_plot(dataset, metric='R2', num_individuals=num_individuals)
    # generate_scores_model_comparison_plot(dataset, metric='MSE', y_min=0, y_max=0.02, suffix='scaled')
    # generate_scores_model_comparison_plot(dataset, metric='R2', y_min=0.95, y_max=1.0, suffix='scaled')
    
    # score evolution plots
    # generate_score_evolution_plots(dataset)
    
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)
    
    # generate_score_evolution_comparison_plot(dataset, metric="MSE", y_max=0.01, suffix='scaled')
    # generate_score_evolution_comparison_plot(dataset, metric="R2", y_min=0.96, y_max=1, suffix='scaled')
