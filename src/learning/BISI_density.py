import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from constants import *
from learning.density_dataset import *
from learning.models.files import generate_model_file
from learning.test.files import generate_scores_file
from learning.test.plots import (generate_score_evolution_comparison_plots,
                                 generate_score_evolution_plots,
                                 generate_scores_model_comparison_plot)
from learning.test.scores import print_scores
from utils import *


def generate_evaluation_plots_dataset11_density():
    dataset = DATASET11_DENSITY
    
    # score model comparison plots
    # generate_scores_model_comparison_plot(dataset, metric='MSE', suffix='2_5000_ind')
    # generate_scores_model_comparison_plot(dataset, metric='R2', suffix='2_5000_ind')

    # score evolution plots
    generate_score_evolution_plots(dataset)
    
    generate_score_evolution_comparison_plots(dataset)


def generate_evaluation_plots_dataset12_density():
    dataset = DATASET12_DENSITY
    
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='MSE')
    generate_scores_model_comparison_plot(dataset, metric='R2')

    # score evolution plots
    generate_score_evolution_plots(dataset)
    
    generate_score_evolution_comparison_plots(dataset)


def generate_evaluation_plots_dataset13_density():
    dataset = DATASET13_DENSITY
    
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='MSE')
    generate_scores_model_comparison_plot(dataset, metric='R2')

    # score evolution plots
    generate_score_evolution_plots(dataset)
    
    generate_score_evolution_comparison_plots(dataset)


def train_models(dataset):
    knn_model = KNeighborsRegressor(n_neighbors=5)
    dtree_model = DecisionTreeRegressor(random_state=SKLEARN_RANDOM_SEED)
    rf_model = RandomForestRegressor(random_state=SKLEARN_RANDOM_SEED)
    nn_model = MLPRegressor(
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
    
    print('---------------------------------')
    print(dataset.capitalize())
    print('---------------------------------')
    print('KNN')
    print('---------')
    generate_model_and_scores_files(knn_model, dataset, 'KNN')
    print('Decision Tree')
    print('---------')
    generate_model_and_scores_files(dtree_model, dataset, 'DecisionTree')
    print('Random Forest')
    print('---------')
    generate_model_and_scores_files(rf_model, dataset, 'RandomForest')
    print('Neural Network')
    print('---------')
    generate_model_and_scores_files(nn_model, dataset, 'NeuralNetwork')


def generate_model_and_scores_files(model, dataset, model_name):
    split = get_dataset_density_train_test_split(dataset, scaled=True)
    X, y, X_train, X_test, y_train, y_test = split
    
    model.fit(X_train, y_train)
    
    # model
    generate_model_file(dataset, model, model_name)
    
    # scores
    y_pred = model.predict(X_test)
    generate_scores_file(X, y, X_test, y_test, y_pred, model, dataset=dataset, model_name=model_name)
    print_scores(dataset, model_name)
