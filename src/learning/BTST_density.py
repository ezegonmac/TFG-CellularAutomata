from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from constants import *
from learning.density_dataset import *
from learning.density_dataset import generate_model_and_scores_files
from learning.models.files import *
from learning.test import *
from learning.test.plots import *
from utils import *


def train_and_test_models8():
    dataset = DATASET8_DENSITY
    
    # Hyperparameters
    
    hp_knn = {}
    
    hp_dtree = {}
    
    rf_model = {}
    
    hp_nn = {
        'hidden_layer_sizes': (3),
        'solver': 'lbfgs',
        'activation': 'tanh',
    }
    
    train_and_test_models(
        dataset,
        hyperparameters_knn=hp_knn,
        hyperparameters_dtree=hp_dtree,
        hyperparameters_rf=rf_model,
        hyperparameters_nn=hp_nn,
        model_variation='vector',
        )
    
    # FIGURES
    # score model comparison plots
    # generate_scores_model_comparison_plot(dataset, metric='MSE')
    # generate_scores_model_comparison_plot(dataset, metric='R2')

    # score evolution plots
    # generate_score_evolution_plots(dataset)
    # generate_score_evolution_comparison_plots(dataset)


def train_and_test_models3():
    dataset = DATASET3_DENSITY
    # train_and_test_models(dataset)
    
    # FIGURES
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='MSE')
    generate_scores_model_comparison_plot(dataset, metric='R2')
    generate_scores_model_comparison_plot(dataset, metric='MSE', y_min=0, y_max=0.02, suffix='scaled')
    generate_scores_model_comparison_plot(dataset, metric='R2', y_min=0.95, y_max=1.0, suffix='scaled')
    
    # score evolution plots
    generate_score_evolution_plots(dataset)
    
    generate_score_evolution_comparison_plots(dataset)

    generate_score_evolution_comparison_plot(dataset, metric="MSE", y_max=0.01, suffix='scaled')
    generate_score_evolution_comparison_plot(dataset, metric="R2", y_min=0.96, y_max=1, suffix='scaled')


def train_and_test_model9():
    dataset = DATASET9_DENSITY
    
    # Hyperparameters
    hp_knn = {
        'n_neighbors': 5,
    }
    
    hp_dtree = {}
    
    rf_model = {}
    
    hp_nn = {
        'early_stopping': True,
        'hidden_layer_sizes': (9, 18, 9),
        'alpha': 0.05,  # not important
        'learning_rate': 'invscaling',  # medium importance
        'max_iter': 3000,
        'solver': 'lbfgs',  # important
        'activation': 'tanh',  # not important
        'batch_size': 300
    }
    
    train_and_test_models(
        dataset,
        hyperparameters_knn=hp_knn,
        hyperparameters_dtree=hp_dtree,
        hyperparameters_rf=rf_model,
        hyperparameters_nn=hp_nn
        )
    
    # FIGURES
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='MSE')
    generate_scores_model_comparison_plot(dataset, metric='R2')
    # generate_scores_model_comparison_plot(dataset, metric='MSE', y_min=0, y_max=0.02, suffix='scaled')
    # generate_scores_model_comparison_plot(dataset, metric='R2', y_min=0.95, y_max=1.0, suffix='scaled')
    
    # score evolution plots
    generate_score_evolution_plots(dataset)
    
    generate_score_evolution_comparison_plots(dataset)

    # generate_score_evolution_comparison_plot(dataset, metric="MSE", y_max=0.01, suffix='scaled')
    # generate_score_evolution_comparison_plot(dataset, metric="R2", y_min=0.96, y_max=1, suffix='scaled')


def train_and_test_models10():
    dataset = DATASET10_DENSITY
    train_and_test_models(dataset)
    
    # FIGURES
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='MSE')
    generate_scores_model_comparison_plot(dataset, metric='R2')
    # generate_scores_model_comparison_plot(dataset, metric='MSE', y_min=0, y_max=0.02, suffix='scaled')
    # generate_scores_model_comparison_plot(dataset, metric='R2', y_min=0.95, y_max=1.0, suffix='scaled')
    
    # score evolution plots
    generate_score_evolution_plots(dataset)
    
    generate_score_evolution_comparison_plots(dataset)

    # generate_score_evolution_comparison_plot(dataset, metric="MSE", y_max=0.01, suffix='scaled')
    # generate_score_evolution_comparison_plot(dataset, metric="R2", y_min=0.96, y_max=1, suffix='scaled')


def train_and_test_models(dataset, model_variation='vector', hyperparameters_knn={}, hyperparameters_dtree={}, hyperparameters_rf= {}, hyperparameters_nn={}):
    
    knn_model = KNeighborsRegressor(**hyperparameters_knn)
    dtree_model = DecisionTreeRegressor(random_state=SKLEARN_RANDOM_SEED, **hyperparameters_dtree)
    rf_model = RandomForestRegressor(random_state=SKLEARN_RANDOM_SEED, **hyperparameters_rf)
    nn_model = MLPRegressor(
        random_state=SKLEARN_RANDOM_SEED,
        # verbose=True
        **hyperparameters_nn,
        )
    
    print('---------------------------------')
    print(dataset.capitalize())
    print('---------------------------------')
    
    print('KNN')
    print('---------')
    generate_model_and_scores_files(knn_model, dataset, 'KNN', model_variation)
    
    print('---------')
    print('Decision Tree')
    print('---------')
    generate_model_and_scores_files(dtree_model, dataset, 'DecisionTree', model_variation)
    
    print('---------')
    print('Random Forest')
    print('---------')
    generate_model_and_scores_files(rf_model, dataset, 'RandomForest', model_variation)
    
    print('---------')
    print('Neural Network')
    print('---------')
    generate_model_and_scores_files(nn_model, dataset, 'NeuralNetwork', model_variation)
