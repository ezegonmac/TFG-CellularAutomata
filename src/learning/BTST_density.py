from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from constants import *
from utils import *
from learning.scoring import *
from learning.test import *
from learning.models import *
from learning.density_dataset import *


def evaluate_dataset8():
    dataset = DATASET8_DENSITY
    # evaluate_dataset(dataset)
    
    # FIGURES
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='MSE')
    generate_scores_model_comparison_plot(dataset, metric='R2')

    # score evolution plots
    generate_score_evolution_plots(dataset)
    generate_score_evolution_comparison_plots(dataset)


def evaluate_dataset3():
    dataset = DATASET3_DENSITY
    # evaluate_dataset(dataset)
    
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


def evaluate_dataset9():
    dataset = DATASET9_DENSITY
    evaluate_dataset(dataset)
    
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


def evaluate_dataset10():
    dataset = DATASET10_DENSITY
    evaluate_dataset(dataset)
    
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


def evaluate_dataset(dataset):
    knn_model = KNeighborsRegressor(n_neighbors=5)
    dtree_model = DecisionTreeRegressor(random_state=SKLEARN_RANDOM_SEED)
    rf_model = RandomForestRegressor(random_state=SKLEARN_RANDOM_SEED)
    # nn_model = MLPRegressor(hidden_layer_sizes=(9, 18, 9), max_iter=500, random_state=SKLEARN_RANDOM_SEED)
    nn_model = MLPRegressor(
        hidden_layer_sizes=(9, 18, 9),
        alpha=0.05,  # not important
        learning_rate='invscaling',  # medium importance
        max_iter=3000,
        solver='lbfgs',  # important
        activation='tanh',  # not important
        
        random_state=SKLEARN_RANDOM_SEED,
        batch_size=300,
        early_stopping=True,
        verbose=True
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
    split = get_dataset_density_train_test_split(dataset, scaled=False)
    X, y, X_train, X_test, y_train, y_test = split
    
    model.fit(X_train, y_train)
    
    # model
    generate_model_file(dataset, model, model_name)
    
    # scores
    y_pred = model.predict(X_test)
    generate_scores_file(X, y, X_test, y_test, y_pred, model, dataset=dataset, model_name=model_name)
    print_scores(dataset, model_name)
