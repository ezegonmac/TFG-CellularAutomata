from constants import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from learning.evaluation import print_evaluation


def main():
    
    knn_model = KNeighborsRegressor(n_neighbors=5)
    
    evaluate_model_with_dataset(knn_model, DATASET8)


def evaluate_model_with_dataset(model, dataset):
    df = load_dataset_density(dataset)
    
    iterations = [str(i) for i in range(1, 10)]
    X = df[['B', 'S', '0']]
    y = df[iterations]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SKLEARN_RANDOM_SEED)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    print_evaluation(X, y, X_test, y_test, y_pred, model)


def load_dataset_density(dataset):
    dataset_name = dataset + '_density'
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    file = f'{dataset_folder}/density_dataset.csv'
    
    df = pd.read_csv(file)
    
    return df
