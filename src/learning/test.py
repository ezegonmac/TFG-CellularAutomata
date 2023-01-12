from constants import *
from learning.scoring import get_mse_by_iteration, get_r2_by_iteration
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

TEST_FIGURES_FOLDER = f'{FIGURES_FOLDER}/test'

def generate_evaluation_figs(y_test, y_pred, dataset, model_name):
    
    mse_by_iteration = get_mse_by_iteration(y_test, y_pred)
    r2_by_iteration = get_r2_by_iteration(y_test, y_pred)
    
    generate_evaluation_fig(mse_by_iteration.values(), dataset, metric="MSE", model_name=model_name)
    generate_evaluation_fig(r2_by_iteration.values(), dataset, metric="R2", model_name=model_name)

def generate_evaluation_fig(scores_by_iteration, dataset, metric, model_name, show=False):

    plot_scores_evolution(scores_by_iteration, dataset=dataset, metric=metric)

    plt.show() if show else plt.savefig(f'{TEST_FIGURES_FOLDER}/{metric}_evolution_{model_name}_{dataset}.png', dpi=300)
    plt.close()

def plot_scores_evolution(scores_evolution, dataset, metric):
    title = f'{dataset.capitalize()} - {metric}'
    
    ax = plt.subplot(111)
    
    color = 'blue' if metric == 'MSE' else 'red'
    ax.plot(scores_evolution, color=color)
    
    ax.set(xlim=(0, 11), ylim=(0, 1.1), xlabel='Iteraciones', ylabel=metric, title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def print_evaluation(X, y, X_test, y_test, y_pred, model):
    # print_random_checks(y_test, y_pred)
    print_overall_score(X_test, y_test, model)
    print_cross_val_score(X, y, model)
    
    mse_by_iteration = get_mse_by_iteration(y_test, y_pred)
    r2_by_iteration = get_r2_by_iteration(y_test, y_pred)
    
    print('MSE by iteration:' + str(mse_by_iteration))
    print('R2 by iteration:' + str(r2_by_iteration))


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
