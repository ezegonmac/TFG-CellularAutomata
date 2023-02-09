import os
from ast import literal_eval

import pandas as pd

from constants import *
from utils import *

from learning.test.scores import (
    calc_RMSE, 
    calc_R2, 
    calc_RMSE_std, 
    calc_R2_std, calc_rmse_by_iteration, 
    calc_r2_by_iteration, 
    calc_rmse_std_by_iteration, 
    calc_r2_std_by_iteration
    )

# LOAD SCORES

def load_scores_file(dataset):
    data_learning_folder = get_data_learning_folder(dataset)
    scores_file = f'{data_learning_folder}/scores.csv'
    df = pd.read_csv(scores_file)
    df['RMSE by iteration'] = df['RMSE by iteration'].apply(lambda x: literal_eval(str(x)))
    df['R2 by iteration'] = df['R2 by iteration'].apply(lambda x: literal_eval(str(x)))
    df['RMSE std by iteration'] = df['RMSE std by iteration'].apply(lambda x: literal_eval(str(x)))
    df['R2 std by iteration'] = df['R2 std by iteration'].apply(lambda x: literal_eval(str(x)))
    return df


def get_scores_by_dataset_and_model(dataset, model_name):
    df = load_scores_file(dataset)
    row = df.loc[(df['Dataset'] == dataset) & (df['Model'] == model_name)]
    return row


def get_scores_by_dataset(dataset):
    df = load_scores_file(dataset)
    rows = df.loc[df['Dataset'] == dataset]
    return rows


# GENERATE SCORES

def generate_scores_file(y_test, y_pred, dataset, model_name, model_variation):
    rmse = calc_RMSE(y_test, y_pred)
    r2 = calc_R2(y_test, y_pred)
    mse_std = calc_RMSE_std(y_test, y_pred)
    r2_std = calc_R2_std(y_test, y_pred)
    rmse_by_iteration = calc_rmse_by_iteration(y_test, y_pred)
    r2_by_iteration = calc_r2_by_iteration(y_test, y_pred)
    rmse_std_by_iteration = calc_rmse_std_by_iteration(y_test, y_pred)
    r2_std_by_iteration = calc_r2_std_by_iteration(y_test, y_pred)

    data = {
        'Dataset': dataset, 
        'Model': model_name, 
        'Model variation': model_variation, 
        'RMSE': rmse, 
        'R2': r2, 
        'RMSE std': mse_std,
        'R2 std': r2_std,
        'RMSE by iteration': str(rmse_by_iteration),
        'R2 by iteration': str(r2_by_iteration),
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
    row = df.loc[(df['Dataset'] == dataset) & (df['Model'] == model_name)]
    if row.empty:
        row = pd.DataFrame.from_dict(data, orient='index').T
        df = pd.concat([df, row], ignore_index=True)
    else:
        for key, value in data.items():
            df.loc[(df['Dataset'] == dataset) & (df['Model'] == model_name), key] = value

    # save file
    df.to_csv(scores_file, index=False)
    print(f'# Scores saved to {scores_file}')
