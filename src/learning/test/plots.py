import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from constants import *
from learning.test.files_scores import (
    get_scores,
    get_scores_by_dataset
    )
from utils import *

MODELS = [KNN, DECISION_TREE, RANDOM_FOREST, NEURAL_NETWORK]


# Score evolution comparison plots

def generate_score_evolution_comparison_plots(dataset, model_variation='vector', num_individuals=500, suffix=''):
    print('# Generating score evolution comparison plots')
    metrics = ['RMSE', 'R2']

    for metric in metrics:
        generate_score_evolution_comparison_plot(dataset, metric=metric, model_variation=model_variation, num_individuals=num_individuals, suffix=suffix)


def generate_score_evolution_comparison_plot(dataset, metric, model_variation='vector', num_individuals=500, suffix='', y_min=0.0, y_max=1.1, show=False):
    print(f'Generating score evolution comparison plot ({dataset} dataset, {metric} metric, {model_variation} model variation, {num_individuals} individuals')
    df = get_scores(dataset, model_variation=model_variation, num_individuals=num_individuals)
    color = 'blue' if metric == 'RMSE' else 'red'

    iterations = len(df[df['Model'] == MODELS[0]][f'{metric} mean by iteration'].values[0].values())
    fig, axs = plt.subplots(1, len(MODELS), figsize=(16, 4), sharey=True)
    
    # Big figure to set common labels
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('Densidad', labelpad=25)
    ax.set_ylabel(metric, labelpad=40)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    
    for i, model in enumerate(MODELS):
        # filter by model
        df_model = df[df['Model'] == model]
        # horizontal line
        axs[i].plot([-1, iterations+0.5], [1, 1], color='black', linewidth=0.35, alpha=0.8, label='_nolegend_')
        # plot
        metric_by_iteration = df_model[f'{metric} mean by iteration'].values[0].values()
        axs[i].plot(metric_by_iteration, color=color)
        # error area
        errors_max = [x + y for x, y in zip(df_model[f'{metric} mean by iteration'].values[0].values(), df[f'{metric} std by iteration'].values[0].values())]
        errors_min = [x - y for x, y in zip(df_model[f'{metric} mean by iteration'].values[0].values(), df[f'{metric} std by iteration'].values[0].values())]
        axs[i].fill_between(range(0, iterations), errors_max, errors_min, alpha=0.30, color=color, label='_nolegend_', linewidth=0)
        # extra
        axs[i].set(xlim=(1, iterations), ylim=(y_min, y_max))
        axs[i].set(title=model)
        axs[i].spines.right.set_visible(False)
        axs[i].spines.top.set_visible(False)
        # ticks fontsize
        axs[i].tick_params(axis='x', labelsize=18)
        axs[i].tick_params(axis='y', labelsize=18)
        # title fontsize
        axs[i].title.set_fontsize(20)
        # integer ticks (if iterations > 10 then only show some ticks)
        if iterations <= 10:
            axs[i].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        else:
            # 5 ticks including min and max
            ticks = np.linspace(1, iterations, 5)
            int_ticks = [round(i) for i in ticks]
            axs[i].set_xticks(int_ticks)

    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_test_figures_folder(dataset)
    if show:
        plt.show()
    else:
        plt.savefig(f'{test_figures_folder}/evolution_comparison_{dataset}_{metric}_{model_variation}_{num_individuals}ind{suffix}.png', dpi=300, bbox_inches='tight')
        print(f'Figure saved in {test_figures_folder}/evolution_comparison_{dataset}_{metric}_{model_variation}_{num_individuals}ind{suffix}.png')
    plt.close()


# Score evolution plots

def generate_score_evolution_plots(dataset, num_individuals=500, suffix=''):
    print('# Generating score evolution plots')
    metrics = ['RMSE', 'R2']
    
    for model in MODELS:
        for metric in metrics:
            generate_score_evolution_plot(dataset, model_name=model, metric=metric, num_individuals=num_individuals, suffix=suffix)


# Score evolution plot

def generate_score_evolution_plot(dataset, model_name, metric, model_variation='vector', num_individuals=500, suffix='', show=False):
    print(f'Generating score evolution plot ({dataset} dataset, {metric} metric, {model_variation} model variation, {num_individuals} individuals')
    plot_score_evolution(dataset, model_name, metric, model_variation, num_individuals)

    test_figures_folder = get_test_figures_folder(dataset)
    file = f'{test_figures_folder}/{metric}_evolution_{model_name}_{dataset}_{num_individuals}ind{suffix}.png'
    if show:
        plt.show()
    else:
        plt.savefig(file, dpi=300)
        print(f'Figure saved in {file}')
    plt.close()


def plot_score_evolution(dataset, model_name, metric, model_variation='vector', num_individuals=500):
    plt.figure(figsize=(5, 5))
    
    df = get_scores(dataset, model_name=model_name, model_variation=model_variation, num_individuals=num_individuals)
    
    scores_evolution = df[f'{metric} mean by iteration'].values[0].values()
    
    title = f'{metric}'
    ax = plt.subplot(111)
    color = 'blue' if metric == 'RMSE' else 'red'
    ax.plot(scores_evolution, color=color)
    
    # top limit line
    iterations = len(scores_evolution)
    ax.set(xlim=(1, iterations), ylim=(0, 1.1), xlabel='Iteraciones', ylabel=metric, title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# Model comparison plots

def generate_score_model_comparison_plots(dataset, model_variation='vector', num_individuals=500, suffix='', y_min=0.0, y_max=1.1, show=False):
    print('# Generating model comparison plots')
    metrics = ['RMSE', 'R2']
    
    for metric in metrics:
        generate_scores_model_comparison_plot(dataset, metric, model_variation, num_individuals, suffix, y_min, y_max, show)


def generate_scores_model_comparison_plot(dataset, metric, model_variation='vector', num_individuals=500, suffix='', y_min=0.0, y_max=1.1, show=False):
    print(f'Generating model comparison plot ({dataset} dataset, {metric} metric, {model_variation} model variation, {num_individuals} individuals')
    metric_mean = f'{metric} mean'
    metric_std = f'{metric} std'

    df = get_scores(dataset, model_variation=model_variation, num_individuals=num_individuals)
    
    # error = 2*std
    df['Double std'] = 2*df[metric_std]
    # filter columns
    df = df[['Model', metric_mean, 'Double std']]
    df = df.set_index(['Model'])
    
    colormap = 'winter' if metric == 'RMSE' else 'autumn'
    df.plot.bar(figsize=(10, 10), width=0.8, colormap=colormap, yerr="Double std", capsize=4)

    plt.xticks(rotation=-30, fontsize=20)
    plt.ylim((y_min, y_max))
    plt.title(f'Comparación de los modelos - {metric}', fontsize=25)
    plt.xlabel('')
    plt.yticks(fontsize=22)
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    # plt.legend(fontsize=16)
    plt.gca().legend_.remove()

    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_test_figures_folder(dataset)
    file = f'{test_figures_folder}/model_comparison_{metric}_{dataset}_{model_variation}_{num_individuals}ind{suffix}.png'
    if show:
        plt.show()
    else:
        plt.savefig(file, dpi=300, bbox_inches='tight')
        print(f'Figure saved in {file}')


def generate_scores_model_individuals_comparison_plot(dataset, metric, model_variation='vector', suffix='', y_min=0.0, y_max=1.1, show=False):
    df = get_scores_by_dataset(dataset)
    # filter by MODELS
    df = df[df['Model'].isin(MODELS)]
    df = df.sort_values(by=[metric])
    # filter by model variation
    df = df[df['Model variation'] == model_variation]
    df = df.groupby(['Model', 'Number of individuals']).mean().reset_index()
    df = df[['Model', 'Number of individuals', metric]]
    # group by num individuals
    df = df.pivot(index='Model', columns='Number of individuals', values=metric)
    
    colormap = 'winter_r' if metric == 'RMSE' else 'autumn_r'
    df.plot.bar(figsize=(10, 10), width=0.8, colormap=colormap)
    # df.plot.line(figsize=(10, 10), colormap=colormap)

    plt.xticks(rotation=0, fontsize=16)
    plt.ylim((y_min, y_max))
    plt.title(f'Comparación de los modelos - {metric}', fontsize=20)
    plt.xlabel('')
    plt.yticks(fontsize=16)
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.legend(fontsize=16)

    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_test_figures_folder(dataset)
    plt.show() if show else plt.savefig(f'{test_figures_folder}/model_comparison_{metric}_{dataset}_{model_variation}_byind{suffix}.png', dpi=300)
