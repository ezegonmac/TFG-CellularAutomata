import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import *
from learning.test import *
from utils import *

TEST_SIZE = 0.2


def load_dataset_density(dataset):
    data_dataset_folder = get_data_datasets_folder(dataset)
    dataset_folder = f'{data_dataset_folder}/{dataset}'
    file = f'{dataset_folder}/density_dataset.csv'
    
    df = pd.read_csv(file)
    
    return df


def get_dataset_density_X_y_split(dataset, num_individuals=None):
    df = load_dataset_density(dataset)
    
    # slice num_individuals
    if num_individuals is not None:
        # mix dataset
        df = df.sample(frac=1, random_state=SKLEARN_RANDOM_SEED)
        # slice num_individuals
        df = df[:num_individuals]
    
    dataset_type = DATASETS_BY_TYPE[dataset]
    
    x_columns = []
    num_iterations = 0
    if dataset_type == 'BTST':
        x_columns = ['B', 'S', '0']
        num_iterations = len(df.columns) - len(x_columns) - 2 # 2 from Unnamed and id
    if dataset_type == 'BISI':
        x_columns = ['Bl', 'Bt', 'Sl', 'St', '0']
        num_iterations = len(df.columns) - len(x_columns) - 4 # from Unnamed, id, B, S 
    
    iterations = [str(i) for i in range(1, num_iterations + 1)]
    X = df[x_columns]
    y = df[iterations]
    
    return X, y


def get_dataset_density_X_y_split_by_fold(dataset, num_individuals, fold):
    df = load_dataset_density(dataset)
    
    # slice fold (num_individuals*fold, num_individuals*(fold+1))
    df = df[num_individuals*fold:num_individuals*(fold+1)]
    
    dataset_type = DATASETS_BY_TYPE[dataset]
    
    x_columns = []
    num_iterations = 0
    if dataset_type == 'BTST':
        x_columns = ['B', 'S', '0']
        num_iterations = len(df.columns) - len(x_columns) - 2 # 2 from Unnamed and id
    if dataset_type == 'BISI':
        x_columns = ['Bl', 'Bt', 'Sl', 'St', '0']
        num_iterations = len(df.columns) - len(x_columns) - 4 # from Unnamed, id, B, S 
    
    iterations = [str(i) for i in range(1, num_iterations + 1)]
    X = df[x_columns]
    y = df[iterations]
    
    return X, y


def get_dataset_density_train_test_split(dataset, scaled=False, num_individuals=None):
    X, y = get_dataset_density_X_y_split(dataset, num_individuals=num_individuals)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SKLEARN_RANDOM_SEED)
    
    if scaled:
        xscaler, yscaler = get_x_y_scalers(dataset)
        
        X_train = pd.DataFrame(xscaler.transform(X_train))
        X_test = pd.DataFrame(xscaler.transform(X_test))
        
        y_train = pd.DataFrame(yscaler.transform(y_train))
        y_test = pd.DataFrame(yscaler.transform(y_test))
    
    return X, y, X_train, X_test, y_train, y_test


def get_dataset_density_train_test_split_by_fold(dataset, fold, scaled=False, num_individuals=None):
    X, y = get_dataset_density_X_y_split_by_fold(dataset, num_individuals=num_individuals, fold=fold)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SKLEARN_RANDOM_SEED)
    
    if scaled:
        xscaler, yscaler = get_x_y_scalers(dataset)
        
        X_train = pd.DataFrame(xscaler.transform(X_train))
        X_test = pd.DataFrame(xscaler.transform(X_test))
        
        y_train = pd.DataFrame(yscaler.transform(y_train))
        y_test = pd.DataFrame(yscaler.transform(y_test))
    
    return X, y, X_train, X_test, y_train, y_test


def get_x_y_scalers(dataset):
    X, y = get_dataset_density_X_y_split(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SKLEARN_RANDOM_SEED)
    
    xscaler = StandardScaler().fit(X_train)
    yscaler = StandardScaler().fit(y_train)
    
    return xscaler, yscaler
