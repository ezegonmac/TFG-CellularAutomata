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
from learning.test.scores_print import print_scores
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


def generate_models_score_plots_ds11(num_individuals):
    dataset = DATASET11_DENSITY
    
    # score model comparison plots
    generate_score_model_comparison_plots(dataset, num_individuals=num_individuals)
    # score evolution plots
    generate_score_evolution_plots(dataset, num_individuals=num_individuals)
    # score evolution comparison plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)


def generate_models_score_plots_ds12(num_individuals):
    dataset = DATASET12_DENSITY
    
    # score model comparison plots
    generate_score_model_comparison_plots(dataset, num_individuals=num_individuals)
    # score evolution plots
    generate_score_evolution_plots(dataset, num_individuals=num_individuals)
    # score evolution comparison plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)


def generate_models_score_plots_ds13(num_individuals):
    dataset = DATASET13_DENSITY
    
    # score model comparison plots
    generate_score_model_comparison_plots(dataset, num_individuals=num_individuals)
    # score evolution plots
    generate_score_evolution_plots(dataset, num_individuals=num_individuals)
    # score evolution comparison plots
    generate_score_evolution_comparison_plots(dataset, num_individuals=num_individuals)
