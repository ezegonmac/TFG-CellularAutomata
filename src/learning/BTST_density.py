import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from constants import *
from learning.scoring import *
from learning.test import *


def evaluate_dataset8():
    dataset = DATASET8
    evaluate_dataset(dataset)
    
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='MSE')
    generate_scores_model_comparison_plot(dataset, metric='R2')

    # score evolution plots
    generate_score_evolution_plots(dataset)
    generate_score_evolution_comparison_plots(dataset)


def evaluate_dataset3():
    dataset = DATASET3
    evaluate_dataset(dataset)
    
    # score model comparison plots
    generate_scores_model_comparison_plot(dataset, metric='MSE')
    generate_scores_model_comparison_plot(dataset, metric='R2')
    generate_scores_model_comparison_plot(dataset, metric='MSE', y_min=0, y_max=0.05, suffix='scaled')
    generate_scores_model_comparison_plot(dataset, metric='R2', y_min=0.95, y_max=1.0, suffix='scaled')
    
    # score evolution plots
    generate_score_evolution_plots(dataset)
    
    generate_score_evolution_comparison_plots(dataset)

    generate_score_evolution_comparison_plot(dataset, metric="MSE", y_max=0.03, suffix='scaled')
    generate_score_evolution_comparison_plot(dataset, metric="R2", y_min=0.96, y_max=0.99, suffix='scaled')


def evaluate_dataset(dataset):
    knn_model = KNeighborsRegressor(n_neighbors=5)
    dtree_model = DecisionTreeRegressor(random_state=SKLEARN_RANDOM_SEED)
    rf_model = RandomForestRegressor(random_state=SKLEARN_RANDOM_SEED)
    nn_model = MLPRegressor(hidden_layer_sizes=(9, 18, 9), max_iter=500, random_state=SKLEARN_RANDOM_SEED)
    
    print('---------------------------------')
    print(dataset.capitalize())
    print('---------------------------------')
    print('KNN')
    print('---------')
    evaluate_model_with_dataset(knn_model, dataset, 'KNN')
    print('Decision Tree')
    print('---------')
    evaluate_model_with_dataset(dtree_model, dataset, 'DecisionTree')
    print('Random Forest')
    print('---------')
    evaluate_model_with_dataset(rf_model, dataset, 'RandomForest')
    print('Neural Network')
    print('---------')
    evaluate_model_with_dataset(nn_model, dataset, 'NeuralNetwork')


def evaluate_model_with_dataset(model, dataset, model_name):
    df = load_dataset_density(dataset)
    
    iterations = [str(i) for i in range(1, 10)]
    X = df[['B', 'S', '0']]
    y = df[iterations]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SKLEARN_RANDOM_SEED)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    generate_scores_file(X, y, X_test, y_test, y_pred, model, dataset=dataset, model_name=model_name)
    print_evaluation(dataset, model_name)


def load_dataset_density(dataset):
    dataset_name = dataset + '_density'
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    file = f'{dataset_folder}/density_dataset.csv'
    
    df = pd.read_csv(file)
    
    return df
