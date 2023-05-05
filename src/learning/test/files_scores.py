import os
from ast import literal_eval

import pandas as pd

from constants import *
from learning.test.files_executions import load_executions_file
from learning.test.scores import calc_score_statistics_by_iteration
from utils import *


# LOAD SCORES

def load_scores_file(dataset):
    data_learning_folder = get_data_learning_folder(dataset)
    scores_file = f'{data_learning_folder}/scores.csv'
    df = pd.read_csv(scores_file)
    df['RMSE mean by iteration'] = df['RMSE mean by iteration'].apply(lambda x: literal_eval(str(x)))
    df['R2 mean by iteration'] = df['R2 mean by iteration'].apply(lambda x: literal_eval(str(x)))
    df['RMSE std by iteration'] = df['RMSE std by iteration'].apply(lambda x: literal_eval(str(x)))
    df['R2 std by iteration'] = df['R2 std by iteration'].apply(lambda x: literal_eval(str(x)))
    return df


def get_scores_by_dataset_and_model(dataset, model_name):
    df = load_scores_file(dataset)
    row = df.loc[(df['Dataset'] == dataset) & (df['Model'] == model_name)]
    return row


def get_scores_by_dataset(dataset):
    df = load_scores_file(dataset)
    # trow error if dataset is empty
    if df.empty:
        raise Exception(f'No scores found for dataset {dataset}. Maybe you have to generate them first.')
    
    rows = df.loc[df['Dataset'] == dataset]
    return rows


# GENERATE SCORES

def generate_scores_file_from_executions(dataset, model_name, model_variation, num_individuals):
    df = load_executions_file(dataset)
    # filter by dataset, model, model variation and number of individuals
    df = df.loc[(df['Dataset'] == dataset) & (df['Model'] == model_name) & (df['Model variation'] == model_variation) & (df['Number of individuals'] == num_individuals)]
    
    # Statistics of all executions
    rmse_mean = df['RMSE'].mean()
    r2_mean = df['R2'].mean()
    rmse_std = df['RMSE'].std()
    r2_std = df['R2'].std()
    
    rmses_by_iteration = df['RMSE by iteration']
    [rmse_mean_by_iteration, rmse_std_by_iteration] = calc_score_statistics_by_iteration(rmses_by_iteration)

    r2s_by_iteration = df['R2 by iteration']
    [r2_mean_by_iteration, r2_std_by_iteration] = calc_score_statistics_by_iteration(r2s_by_iteration)

    data = {
        'Dataset': dataset, 
        'Model': model_name, 
        'Model variation': model_variation, 
        'Number of individuals': num_individuals, 
        'RMSE mean': rmse_mean, 
        'R2 mean': r2_mean, 
        'RMSE std': rmse_std,
        'R2 std': r2_std,
        'RMSE mean by iteration': str(rmse_mean_by_iteration),
        'R2 mean by iteration': str(r2_mean_by_iteration),
        'RMSE std by iteration': str(rmse_std_by_iteration),
        'R2 std by iteration': str(r2_std_by_iteration)
    }

    # create csv file if not exists
    data_learning_folder = get_data_learning_folder(dataset)
    scores_file = f'{data_learning_folder}/scores.csv'
    if not os.path.isfile(scores_file):
        df = pd.DataFrame.from_dict(data, orient='index').T
    else:
        df = pd.read_csv(scores_file)

    # add data as row to csv file, or update if exists
    row = df.loc[(df['Dataset'] == dataset) & (df['Model'] == model_name) & (df['Model variation'] == model_variation) & (df['Number of individuals'] == num_individuals)]
    if row.empty:
        row = pd.DataFrame.from_dict(data, orient='index').T
        df = pd.concat([df, row], ignore_index=True)
    else:
        for key, value in data.items():
            df.loc[(df['Dataset'] == dataset) & (df['Model'] == model_name), key] = value

    # save file
    df.to_csv(scores_file, index=False)
    print(f'# Scores saved to {scores_file}')
