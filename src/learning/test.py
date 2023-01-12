from constants import *
from learning.scoring import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

TEST_FIGURES_FOLDER = f'{FIGURES_FOLDER}/test'


def generate_evaluation_figs(y_test, y_pred, dataset, model_name):
    generate_evaluation_fig(dataset, model_name=model_name, metric="MSE")
    generate_evaluation_fig(dataset, model_name=model_name, metric="R2")


def generate_evaluation_fig(dataset, model_name, metric, show=False):
    plot_scores_evolution(dataset, model_name, metric)

    plt.show() if show else plt.savefig(f'{TEST_FIGURES_FOLDER}/{metric}_evolution_{model_name}_{dataset}.png', dpi=300)
    plt.close()


def plot_scores_evolution(dataset, model_name, metric):
    df = get_scores_by_dataset_and_model(dataset, model_name)
    scores_evolution = df[f'{metric} by iteration'].values[0].values()
    
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


def print_evaluation(dataset, model_name):
    row = get_scores_by_dataset_and_model(dataset, model_name)
    
    print("MSE: " + str(row['MSE'].values[0]))
    print("R2: " + str(row['R2'].values[0]))
    print("CV MSE: " + str(row['CV MSE'].values[0]))
    print("CV R2: " + str(row['CV R2'].values[0]))
    print("MSE by iteration: " + str(row['MSE by iteration'].values[0]))
    print("R2 by iteration: " + str(row['R2 by iteration'].values[0]))


def print_random_checks(y_test, y_pred, iteration='1'):
    random_idx = np.random.choice(y_test.shape[0], 10)
    predict_df = pd.DataFrame()
    
    predict_df['Real'] = y_test.iloc[random_idx][iteration]
    predict_df['Predict'] = y_pred[random_idx, 0]
    predict_df['Error'] = predict_df['Real'] - predict_df['Predict']
    
    print('Some random checks:')
    print(predict_df)
