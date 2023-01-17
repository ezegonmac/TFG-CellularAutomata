import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from constants import *
from learning.scoring import *
from utils import *
from learning.density_dataset import *
from learning.models import *

MODELS = [KNN, DECISION_TREE, RANDOM_FOREST, NEURAL_NETWORK]


def generate_score_evolution_comparison_plots(dataset, suffix=''):
    metrics = ['MSE', 'R2']

    for model in MODELS:
        for metric in metrics:
            generate_score_evolution_comparison_plot(dataset, metric=metric, suffix=suffix)


def generate_score_evolution_comparison_plot(dataset, metric, suffix='', y_min=0, y_max=1.1, show=False):
    df = get_scores_by_dataset(dataset)
    color = 'blue' if metric == 'MSE' else 'red'

    iterations = len(df[df['Model'] == MODELS[0]][f'{metric} by iteration'].values[0].values())
    fig, axs = plt.subplots(1, len(MODELS), figsize=(18, 5))
    for i, model in enumerate(MODELS):
        axs[i].plot(df[df['Model'] == model][f'{metric} by iteration'].values[0].values(), color=color)
        axs[i].set(xlim=(1, iterations), ylim=(y_min, y_max))
        axs[i].set(xlabel='Iteraciones', ylabel=metric, title=model)
        axs[i].spines.right.set_visible(False)
        axs[i].spines.top.set_visible(False)

    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_test_figures_folder(dataset)
    plt.show() if show else plt.savefig(f'{test_figures_folder}/{metric}_evolution_comparison_{dataset}{suffix}.png', dpi=300)
    plt.close()


def generate_score_evolution_plots(dataset):
    metrics = ['MSE', 'R2']
    
    for model in MODELS:
        for metric in metrics:
            generate_score_evolution_plot(dataset, model_name=model, metric=metric)


def generate_score_evolution_plot(dataset, model_name, metric, show=False):
    plot_score_evolution(dataset, model_name, metric)

    test_figures_folder = get_test_figures_folder(dataset)
    plt.show() if show else plt.savefig(f'{test_figures_folder}/{metric}_evolution_{model_name}_{dataset}.png', dpi=300)
    plt.close()


def plot_score_evolution(dataset, model_name, metric):
    plt.figure(figsize=(5, 5))
    df = get_scores_by_dataset_and_model(dataset, model_name)
    scores_evolution = df[f'{metric} by iteration'].values[0].values()
    
    title = f'{metric}'
    ax = plt.subplot(111)
    color = 'blue' if metric == 'MSE' else 'red'
    ax.plot(scores_evolution, color=color)
    
    iterations = len(scores_evolution)
    ax.set(xlim=(1, iterations), ylim=(0, 1.1), xlabel='Iteraciones', ylabel=metric, title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def generate_scores_model_comparison_plot(dataset, metric, suffix='', y_min=0, y_max=1.1, show=False):
    df = get_scores_by_dataset(dataset)
    # filter by MODELS
    df = df[df['Model'].isin(MODELS)]
    df = df.sort_values(by=[metric])
    df = df[['Model', metric]]
    df = df.set_index(['Model'])
    
    color = 'blue' if metric == 'MSE' else 'red'
    df.plot.bar(figsize=(10, 10), width=0.8, color=color)

    plt.xticks(rotation=0, fontsize=16)
    plt.ylim((y_min, y_max))
    plt.title(f'Comparación de los modelos - {metric}', fontsize=20)
    plt.legend(fontsize=16)

    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_test_figures_folder(dataset)
    plt.show() if show else plt.savefig(f'{test_figures_folder}/model_comparison_{metric}_{dataset}{suffix}.png', dpi=300)


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
