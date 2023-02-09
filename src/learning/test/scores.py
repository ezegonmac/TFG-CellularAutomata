import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Overall scores

def calc_RMSE(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred, squared=False)
    return mse


def calc_R2(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    return r2


# Standard deviations

def calc_RMSE_std(y_test, y_pred):
    # calc standard deviation of the RMSE
    individuals = y_test.shape[0]
    rmses = []
    for i in range(individuals):
        if y_test.ndim == 1:
            y_test_i = y_test.iloc[i].reshape(-1,1)
            y_pred_i = y_pred[i].reshape(-1,1)
        else:
            y_test_i = y_test.iloc[i, :]
            y_pred_i = y_pred[i, :]
        rmse = calc_RMSE(y_test_i, y_pred_i)
        rmses.append(rmse)
    
    return np.std(rmses)


def calc_R2_std(y_test, y_pred):
    # TODO: meaning of the R2 std ?
    return 0
    
    # calc standard deviation of the RMSE
    individuals = y_test.shape[0]
    r2s = []
    for i in range(individuals):
        if y_test.ndim == 1:
            y_test_i = y_test.iloc[i].reshape(-1,1)
            y_pred_i = y_pred[i].reshape(-1,1)
        else:
            y_test_i = y_test.iloc[i, :]
            y_pred_i = y_pred[i, :]
        r2 = calc_R2(y_test_i, y_pred_i)
        r2s.append(r2)
    
    return np.std(r2s)


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
