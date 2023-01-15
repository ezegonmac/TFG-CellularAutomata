from joblib import dump, load

from constants import *
from utils import *


def generate_model_file(dataset, model, model_name):
    data_learning_folder = get_data_learning_folder(dataset)
    model_folder = f'{data_learning_folder}/{dataset}/models'
    model_file = f'{model_folder}/{model_name}_{dataset}.pkl'

    create_folder_if_not_exists(model_folder)
    
    dump(model, model_file)


def load_model_from_file(dataset, model_name):
    data_learning_folder = get_data_learning_folder(dataset)
    model_folder = f'{data_learning_folder}/{dataset}/models'
    model_file = f'{model_folder}/{model_name}_{dataset}.pkl'

    create_folder_if_not_exists(model_folder)
    
    model = load(model_file)
    
    return model
