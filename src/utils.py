from constants import *
import os


def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# data

def get_data_datasets_folder(dataset):
    dataset_type = DATASETS_BY_TYPE[dataset]
    data_datasets_folder = f'{DATA_DATASETS_FOLDER}/{dataset_type}'
    create_folder_if_not_exists(data_datasets_folder)
    return data_datasets_folder


def get_data_learning_folder(dataset):
    dataset_type = DATASETS_BY_TYPE[dataset]
    data_learning_folder = f'{DATA_LEARNING_FOLDER}/{dataset_type}'
    create_folder_if_not_exists(data_learning_folder)
    return data_learning_folder


def get_data_learning_executions_folder(dataset):
    data_learning_folder = get_data_learning_folder(dataset)
    executions_folder = f'{data_learning_folder}/{dataset}/executions'
    create_folder_if_not_exists(data_learning_folder)
    return executions_folder

# figures

def get_figures_folder(dataset):
    dataset_type = DATASETS_BY_TYPE[dataset]
    figures_folder = f'{FIGURES_FOLDER}/{dataset_type}'
    create_folder_if_not_exists(figures_folder)
    return figures_folder


def get_test_figures_folder(dataset):
    figures_folder = get_figures_folder(dataset)
    test_figures_folder = f'{figures_folder}/test'
    create_folder_if_not_exists(test_figures_folder)
    return test_figures_folder


def get_density_figures_folder(dataset):
    figures_folder = get_figures_folder(dataset)
    density_figures_folder = f'{figures_folder}/statistics/density'
    create_folder_if_not_exists(density_figures_folder)
    return density_figures_folder


def get_chaotic_figures_folder(dataset):
    figures_folder = get_figures_folder(dataset)
    density_figures_folder = f'{figures_folder}/statistics/chaotic'
    create_folder_if_not_exists(density_figures_folder)
    return density_figures_folder


def get_states_figures_folder(dataset):
    figures_folder = get_figures_folder(dataset)
    states_figures_folder = f'{figures_folder}/statistics/states'
    create_folder_if_not_exists(states_figures_folder)
    return states_figures_folder
