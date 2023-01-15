import pandas as pd
from sklearn.model_selection import train_test_split

from constants import *
from learning.test import *
from utils import *


def load_dataset_density(dataset):
    data_dataset_folder = get_data_datasets_folder(dataset)
    dataset_folder = f'{data_dataset_folder}/{dataset}'
    file = f'{dataset_folder}/density_dataset.csv'
    
    df = pd.read_csv(file)
    
    return df


def get_dataset_density_X_y_split(dataset):
    df = load_dataset_density(dataset)
    
    dataset_type = DATASETS_BY_TYPE[dataset]
    x_columns = ['B', 'S', '0'] if dataset_type == 'BTST' else None
    x_columns = ['Bl', 'Bt', 'Sl', 'St', '0'] if dataset_type == 'BISI' else None
    
    num_iterations = len(df.columns) - len(x_columns) - 2
    iterations = [str(i) for i in range(1, num_iterations-1)]
    X = df[x_columns]
    y = df[iterations]
    
    return X, y


def get_dataset_density_train_test_split(dataset):
    X, y = get_dataset_density_X_y_split(dataset)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SKLEARN_RANDOM_SEED)
    
    return X, y, X_train, X_test, y_train, y_test
