from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def get_mse_by_iteration(y_test, y_pred):
    scores = get_cross_val_score_by_iteration(y_test, y_pred, scorer=mean_squared_error)
    return scores


def get_r2_by_iteration(y_test, y_pred):
    scores = get_cross_val_score_by_iteration(y_test, y_pred, scorer=r2_score)
    return scores


def get_score_by_iteration(y_test, y_pred, scorer=mean_squared_error):
    scores_by_iteration = {}
    iterations = y_test.shape[1]
    for i in range(iterations):
        score = scorer(y_test.iloc[:, i], y_pred[:, i])
        scores_by_iteration[str(i+1)] = score
    return scores_by_iteration


def get_cross_val_score_by_iteration(y_test, y_pred, scorer=mean_squared_error, folds=5):
    scores_by_iteration = {}
    iterations = y_test.shape[1]
    for i in range(iterations):
        y_test_i = y_test.iloc[:, i]
        y_pred_i = y_pred[:, i]
        
        folds_score = []
        for fold in range(folds):
            fold_size = len(y_test_i) // folds
            start = fold * fold_size
            end = start + fold_size
            print(f'Start: {start}, end: {end}')
            y_test_fold = y_test_i.iloc[start:end]
            y_pred_fold = y_pred_i[start:end]
            score = scorer(y_test_fold, y_pred_fold)
            folds_score.append(score)
    
        scores_by_iteration[str(i+1)] = np.mean(folds_score)

    return scores_by_iteration
