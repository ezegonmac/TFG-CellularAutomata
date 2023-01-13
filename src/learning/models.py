import os

from joblib import dump, load

from constants import *


def generate_model_file(dataset, model, model_name):
    model_folder = f'{DATA_LEARNING_FOLDER}/{dataset}/models'
    model_file = f'{model_folder}/{model_name}_{dataset}.pkl'
    # generate folder if not exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    dump(model, model_file)


def load_model_from_file(dataset, model_name):
    model_folder = f'{DATA_LEARNING_FOLDER}/{dataset}/models'
    model_file = f'{model_folder}/{model_name}_{dataset}.pkl'
    # generate folder if not exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    model = load(model_file)
    
    return model
