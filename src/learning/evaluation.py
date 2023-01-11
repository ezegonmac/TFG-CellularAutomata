from learning.scoring import get_mse_by_iteration, get_r2_by_iteration
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


def print_evaluation(X, y, X_test, y_test, y_pred, model):
    print_random_checks(y_test, y_pred)
    print_overall_score(X_test, y_test, model)
    print_cross_val_score(X, y, model)
    
    mse_by_iteration = get_mse_by_iteration(y_test, y_pred)
    score_by_iteration = get_r2_by_iteration(y_test, y_pred)
    
    print('MSE by iteration:' + str(mse_by_iteration))
    print('R2 by iteration:' + str(score_by_iteration))


def print_cross_val_score(X, y, model):
    scores = cross_val_score(model, X, y, cv=10)
    
    print(f'Cross validation scores: {scores}')
    print(f'Cross validation mean score: {scores.mean()}')


def print_overall_score(X_test, y_test, model):
    score = model.score(X_test, y_test)
    
    print(f'Score: {score}')


def print_random_checks(y_test, y_pred, iteration='1'):
    random_idx = np.random.choice(y_test.shape[0], 10)
    predict_df = pd.DataFrame()
    
    predict_df['Real'] = y_test.iloc[random_idx][iteration]
    predict_df['Predict'] = y_pred[random_idx, 0]
    predict_df['Error'] = predict_df['Real'] - predict_df['Predict']
    
    print('Some random checks:')
    print(predict_df)
