import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Overall scores

def calc_RMSE(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred, squared=False)
    return mse


def calc_R2(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    return r2


# By iteration scores

def calc_rmse_by_iteration(y_test, y_pred):
    scores = _calc_score_by_iteration(y_test, y_pred, scorer='RMSE')
    return scores


def calc_r2_by_iteration(y_test, y_pred):
    scores = _calc_score_by_iteration(y_test, y_pred, scorer='R2')
    return scores


def _calc_score_by_iteration(y_test, y_pred, scorer='RMSE'):
    scores_by_iteration = {}
    iterations = y_test.shape[1]
    score = None
    for i in range(iterations):
        y_test_i = y_test.iloc[:, i]
        y_pred_i = y_pred[:, i]
        if scorer == 'RMSE':
            score = calc_RMSE(y_test_i, y_pred_i)
        elif scorer == 'R2':
            score = calc_R2(y_test_i, y_pred_i)
        else:
            raise ValueError(f'Unknown score: {score}')
        scores_by_iteration[str(i+1)] = score
    return scores_by_iteration


# By iteration standard deviations

def calc_rmse_std_by_iteration(y_test, y_pred):
    scores = _calc_score_std_by_iteration(y_test, y_pred, scorer='RMSE')
    return scores


def calc_r2_std_by_iteration(y_test, y_pred):
    scores = _calc_score_std_by_iteration(y_test, y_pred, scorer='R2')
    return scores


def _calc_score_std_by_iteration(y_test, y_pred, scorer='RMSE'):
    scores_by_iteration = {}
    iterations = y_test.shape[1]
    score = None
    for i in range(iterations):
        y_test_i = y_test.iloc[:, i].T
        y_pred_i = y_pred[:, i].T
        if scorer == 'RMSE':
            score = calc_RMSE_std(y_test_i, y_pred_i)
        elif scorer == 'R2':
            score = calc_R2_std(y_test_i, y_pred_i)
        else:
            raise ValueError(f'Unknown score: {score}')
        scores_by_iteration[str(i+1)] = score
    return scores_by_iteration



# RMSE and R2 (scores) statistics

def calc_score_statistics_by_iteration(scores_by_iteration):
    executions = scores_by_iteration.shape[0]
    scores_by_iteration = scores_by_iteration.tolist()
    iterations = len(scores_by_iteration[0])
    
    # dict with iteration as keys and values as scores
    values_per_iteration = {}
    for i in range(1, iterations+1):
        values = []
        for e in range(executions):
            execution = scores_by_iteration[e]
            value = execution[str(i)]
            values.append(value)
            
        values_per_iteration[i] = values
    
    score_mean_by_iteration = {}
    for i in range(1, iterations+1):
        score_mean_by_iteration[i] = np.mean(values_per_iteration[i])
    
    score_std_by_iteration = {}
    for i in range(1, iterations+1):
        score_std_by_iteration[i] = np.std(values_per_iteration[i])

    return [score_mean_by_iteration, score_std_by_iteration]
