import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from learning.test.files import get_scores_by_dataset_and_model

# Print all scores

def print_scores(dataset, model_name):
    row = get_scores_by_dataset_and_model(dataset, model_name)
    
    print("MSE: " + str(row['MSE'].values[0]))
    print("R2: " + str(row['R2'].values[0]))
    print("CV MSE: " + str(row['CV MSE'].values[0]))
    print("CV R2: " + str(row['CV R2'].values[0]))
    print("MSE by iteration: " + str(row['MSE by iteration'].values[0]))
    print("R2 by iteration: " + str(row['R2 by iteration'].values[0]))


# Overall scores

def get_overall_MSE(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return mse


def get_overall_R2(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    return r2


# By iteration scores

def get_mse_by_iteration(y_test, y_pred):
    scores = _get_score_by_iteration(y_test, y_pred, scorer=mean_squared_error)
    return scores


def get_r2_by_iteration(y_test, y_pred):
    scores = _get_score_by_iteration(y_test, y_pred, scorer=r2_score)
    return scores


def _get_score_by_iteration(y_test, y_pred, scorer=mean_squared_error):
    scores_by_iteration = {}
    iterations = y_test.shape[1]
    for i in range(iterations):
        score = scorer(y_test.iloc[:, i], y_pred[:, i])
        scores_by_iteration[str(i+1)] = score
    return scores_by_iteration


# Cross val scores

# Cross val score not right implemented
# CVS is for validation, not testing?
# Taking only test data in account
def get_cross_val_score_by_iteration(y_test, y_pred, scorer=mean_squared_error, folds=5):
    scores_by_iteration = {}
    iterations = y_test.shape[1]
    for i in range(iterations):
        y_test_i = y_test.iloc[:, i]
        y_pred_i = y_pred[:, i]
        
        iteration_score = _get_cross_val_score_of_iteration(y_test_i, y_pred_i, scorer, folds)
    
        scores_by_iteration[str(i+1)] = iteration_score

    return scores_by_iteration


def _get_cross_val_score_of_iteration(y_test_i, y_pred_i, scorer, folds):
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
