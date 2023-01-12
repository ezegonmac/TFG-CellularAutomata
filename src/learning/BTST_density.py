import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from constants import *
from learning.scoring import *
from learning.test import *


def main():
    
    evaluate_dataset(DATASET8)
    evaluate_dataset(DATASET3)
    
    generate_scores_model_comparation_plot(DATASET8)
    generate_scores_model_comparation_plot(DATASET3, y_min=0.95, y_max=1.0)


def evaluate_dataset(dataset):
    knn_model = KNeighborsRegressor(n_neighbors=5)
    dtree_model = DecisionTreeRegressor(random_state=0)
    rf_model = RandomForestRegressor(random_state=0)
    
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


def evaluate_model_with_dataset(model, dataset, model_name):
    df = load_dataset_density(dataset)
    
    iterations = [str(i) for i in range(1, 10)]
    X = df[['B', 'S', '0']]
    y = df[iterations]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SKLEARN_RANDOM_SEED)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    # generate_scores_file(X, y, X_test, y_test, y_pred, model, dataset=dataset, model_name=model_name)
    print_evaluation(dataset, model_name)
    generate_evaluation_plots(y_test, y_pred, dataset, model_name=model_name)


def load_dataset_density(dataset):
    dataset_name = dataset + '_density'
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    file = f'{dataset_folder}/density_dataset.csv'
    
    df = pd.read_csv(file)
    
    return df
