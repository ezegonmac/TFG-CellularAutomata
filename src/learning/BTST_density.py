from constants import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from learning.test import print_evaluation, generate_evaluation_figs


def main():
    
    evaluate_dataset(DATASET8)
    evaluate_dataset(DATASET3)

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

    print_evaluation(X, y, X_test, y_test, y_pred, model)
    generate_evaluation_figs(y_test, y_pred, dataset, model_name=model_name)


def load_dataset_density(dataset):
    dataset_name = dataset + '_density'
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    file = f'{dataset_folder}/density_dataset.csv'
    
    df = pd.read_csv(file)
    
    return df
