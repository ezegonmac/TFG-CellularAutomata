import os
from ast import literal_eval

import pandas as pd

from constants import *
from utils import *

from learning.test.scores import (
    calc_RMSE, 
    calc_R2, 
    calc_rmse_by_iteration, 
    calc_r2_by_iteration,
    )
from learning.density_dataset import get_dataset_density_train_test_split_by_fold
from learning.models.files import generate_model_file


# LOAD EXECUTIONS

def load_executions_file(dataset):
    executions_folder = get_data_learning_executions_folder(dataset)
    executions_file = f'{executions_folder}/scores.csv'
    df = pd.read_csv(executions_file)
    df['RMSE by iteration'] = df['RMSE by iteration'].apply(lambda x: literal_eval(str(x)))
    df['R2 by iteration'] = df['R2 by iteration'].apply(lambda x: literal_eval(str(x)))
    return df


# GENERATE EXECUTIONS

def generate_execution_scores_file(model, dataset, model_name, model_variation, num_individuals, num_executions, save_model):
    # drop old rows if exists
    executions_folder = get_data_learning_executions_folder(dataset)
    executions_file = f'{executions_folder}/scores.csv'
    if os.path.isfile(executions_file):
        df = pd.read_csv(executions_file)
        index = df[(df['Model'] == model_name) & (df['Dataset'] == dataset) & (df['Model variation'] == model_variation) & (df['Number of individuals'] == num_individuals)].index
        print(model_name)
        print(df[(df['Model'] == model_name)])
        df.drop(index, inplace=True)
        print(df)
        df.to_csv(executions_file, index=False)
        print(df)
    
    # add one row for each execution
    for execution in range(num_executions):
        print(f' - Execution {execution+1} of {num_executions}')
        
        dataset_fold = execution
        split = get_dataset_density_train_test_split_by_fold(dataset, fold=dataset_fold, scaled=True, num_individuals=num_individuals)
        X, y, X_train, X_test, y_train, y_test = split

        model.fit(X_train, y_train)
        
        # model
        if save_model:
            generate_model_file(dataset, 
                                model, 
                                model_name, 
                                model_variation, 
                                num_individuals
                                )
        
        # scores
        y_pred = model.predict(X_test)
    
        add_execution_scores_row(
            y_test, y_pred,
            dataset=dataset,
            model_name=model_name,
            model_variation=model_variation,
            num_individuals=num_individuals,
            execution=execution,
            )


def add_execution_scores_row(y_test, y_pred, dataset, model_name, model_variation, num_individuals, execution):
    rmse = calc_RMSE(y_test, y_pred)
    r2 = calc_R2(y_test, y_pred)
    rmse_by_iteration = calc_rmse_by_iteration(y_test, y_pred)
    r2_by_iteration = calc_r2_by_iteration(y_test, y_pred)

    data = {
        'Dataset': dataset, 
        'Model': model_name, 
        'Model variation': model_variation, 
        'Number of individuals': num_individuals, 
        'Execution': execution,
        'RMSE': rmse, 
        'R2': r2, 
        'RMSE by iteration': str(rmse_by_iteration),
        'R2 by iteration': str(r2_by_iteration),
    }

    # create csv file if not exists
    executions_folder = get_data_learning_executions_folder(dataset)
    executions_file = f'{executions_folder}/scores.csv'
    if not os.path.isfile(executions_file):
        df = pd.DataFrame.from_dict(data, orient='index').T
    else:
        df = pd.read_csv(executions_file)
    
    # add data as row to csv file, or update if exists
    old_row = df.loc[(df['Dataset'] == dataset) & (df['Model'] == model_name) & (df['Model variation'] == model_variation) & (df['Number of individuals'] == num_individuals) & (df['Execution'] == execution)]
    row = pd.DataFrame.from_dict(data, orient='index').T
    if old_row.empty:
        df = pd.concat([df, row], ignore_index=True)
    else:
        df.update(row)

    # save file
    df.to_csv(executions_file, index=False)
    print(f'# Scores saved to {executions_file}')
