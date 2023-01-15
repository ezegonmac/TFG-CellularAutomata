import os
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from constants import *
from utils import *


def get_cross_val_MSE(X, y, model):
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    return scores.mean()


def get_cross_val_R2(X, y, model):
    scores = cross_val_score(model, X, y, cv=10, scoring='r2')
    return scores.mean()


def get_overall_MSE(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return mse


def get_overall_R2(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    return r2


def get_mse_by_iteration(y_test, y_pred):
    scores = get_score_by_iteration(y_test, y_pred, scorer=mean_squared_error)
    return scores


def get_r2_by_iteration(y_test, y_pred):
    scores = get_score_by_iteration(y_test, y_pred, scorer=r2_score)
    return scores


def get_score_by_iteration(y_test, y_pred, scorer=mean_squared_error):
    scores_by_iteration = {}
    iterations = y_test.shape[1]
    for i in range(iterations):
        score = scorer(y_test.iloc[:, i], y_pred[:, i])
        scores_by_iteration[str(i+1)] = score
    return scores_by_iteration


def generate_scores_file(X, y, X_test, y_test, y_pred, model, dataset, model_name):
    mse = get_overall_MSE(y_test, y_pred)
    cv_mse = get_cross_val_MSE(X, y, model)
    r2 = get_overall_R2(y_test, y_pred)
    cv_r2 = get_cross_val_R2(X, y, model)
    mse_by_iteration = get_mse_by_iteration(y_test, y_pred)
    r2_by_iteration = get_r2_by_iteration(y_test, y_pred)

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


def get_scores_by_model(model_name):
    df = load_scores_file(dataset)
    rows = df.loc[df['Model'] == model_name]
    return rows


# Cross val score not right implemented
# CVS is for validation, not testing?
# Taking only test data in account
def get_cross_val_score_by_iteration(y_test, y_pred, scorer=mean_squared_error, folds=5):
    scores_by_iteration = {}
    iterations = y_test.shape[1]
    for i in range(iterations):
        y_test_i = y_test.iloc[:, i]
        y_pred_i = y_pred[:, i]
        
        iteration_score = get_cross_val_score_of_iteration(y_test_i, y_pred_i, scorer, folds)
    
        scores_by_iteration[str(i+1)] = iteration_score

    return scores_by_iteration


def get_cross_val_score_of_iteration(y_test_i, y_pred_i, scorer, folds):
    folds_score = []
    for fold in range(folds):
        fold_size = len(y_test_i) // folds
        start = fold * fold_size
        end = start + fold_size
        y_test_fold = y_test_i.iloc[start:end]
        y_pred_fold = y_pred_i[start:end]
        score = scorer(y_test_fold, y_pred_fold)
        folds_score.append(score)
    iteration_score = np.mean(folds_score)
    
    return iteration_score
