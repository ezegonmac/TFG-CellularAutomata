import os
from ast import literal_eval

import pandas as pd

from constants import *
from utils import *

from learning.test.scores import get_overall_MSE, get_overall_R2, get_mse_by_iteration, get_r2_by_iteration

# LOAD SCORES

def load_scores_file(dataset):
    data_learning_folder = get_data_learning_folder(dataset)
    scores_file = f'{data_learning_folder}/scores.csv'
    df = pd.read_csv(scores_file)
    df['MSE by iteration'] = df['MSE by iteration'].apply(lambda x: literal_eval(str(x)))
    df['R2 by iteration'] = df['R2 by iteration'].apply(lambda x: literal_eval(str(x)))
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

def generate_scores_file(X, y, X_test, y_test, y_pred, model, dataset, model_name):
    mse = get_overall_MSE(y_test, y_pred)
    # cv_mse = get_cross_val_MSE(X, y, model)
    r2 = get_overall_R2(y_test, y_pred)
    # cv_r2 = get_cross_val_R2(X, y, model)
    mse_by_iteration = get_mse_by_iteration(y_test, y_pred)
    r2_by_iteration = get_r2_by_iteration(y_test, y_pred)
    cv_mse = None
    cv_r2 = None

    data = {
        'Dataset': dataset, 
        'Model': model_name, 
        'MSE': mse, 
        'R2': r2, 
        'CV MSE': cv_mse, 
        'CV R2': cv_r2,
        'MSE by iteration': str(mse_by_iteration),
        'R2 by iteration': str(r2_by_iteration),
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
    print(f'Scores saved to {scores_file}')
