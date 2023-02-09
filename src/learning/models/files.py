from joblib import dump, load

from constants import *
from utils import *


def generate_model_file(dataset, model, model_name, model_variation):
    data_learning_folder = get_data_learning_folder(dataset)
    model_folder = f'{data_learning_folder}/{dataset}/models'
    model_file = f'{model_folder}/{model_name}_{model_variation}_{dataset}.pkl'

    create_folder_if_not_exists(model_folder)
    
    dump(model, model_file)
    print(f'# Model saved to {model_file} - {model_variation}.')


def load_model_from_file(dataset, model_name, model_variation):
    data_learning_folder = get_data_learning_folder(dataset)
    model_folder = f'{data_learning_folder}/{dataset}/models'
    model_file = f'{model_folder}/{model_name}_{model_variation}_{dataset}.pkl'

    create_folder_if_not_exists(model_folder)
    print(model_file)
    model = load(model_file)
    
    return model
