import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from constants import *
from learning.test.files_scores import get_scores_by_rule_type
from utils import get_data_learning_folder_by_rule_type


MODELS = ['KNN', 'DecisionTree', 'RandomForest', 'NeuralNetwork']
BTST_DATASETS = ['dataset3_density', 'dataset9_density', 'dataset10_density']
BISI_DATASETS = ['dataset11_density', 'dataset12_density', 'dataset13_density']
BS_DATASETS = ['dataset14_density', 'dataset15_density', 'dataset16_density']
DATASETS_DICT = {
    'BTST': BTST_DATASETS,
    'BISI': BISI_DATASETS,
    'BS': BS_DATASETS
}

def generate_rule_type_comparation_plot(rule_type='BS', metric='RMSE', model_variation='vector', num_individuals=1000, suffix='', show=False, y_min=0, y_max=1):
    print(f'Generating rule type comparison plot (rule type={rule_type})')
    metric_mean = f'{metric} mean'
    metric_std = f'{metric} std'

    df = get_scores_by_rule_type(rule_type)
    datasets = DATASETS_DICT[rule_type]
    
    if df.empty:
        raise Exception('No scores for this dataset')
    # filter by MODELS
    if df.empty:
        raise Exception('No valid models found')
    # filter by model variation
    if model_variation:
        df = df[df['Model variation'] == model_variation]
        if df.empty:
            raise Exception('No scores for this model variation')
    # filter by num individuals
    if num_individuals:
        df = df[df['Number of individuals'] == num_individuals]
        if df.empty:
            raise Exception('No scores for this number of individuals')
    # filter by datasets
    df = df[df['Dataset'].isin(datasets)]
    if df.empty:
        raise Exception('No scores for this rule type datasets')
    
    print(df)
    print(df.columns)
    
    # error = 2*std
    df['Double std'] = 2*df[metric_std]
    # filter columns
    df = df[['Dataset', metric_mean, 'Model', 'Double std']]
    df = df.set_index(['Dataset'])
    
    print(df)
    print(df.columns)
    
    colormap = 'winter' if metric == 'RMSE' else 'autumn'
    # df.plot.bar(figsize=(10, 10), width=0.8, colormap=colormap, yerr="Double std", capsize=4)
    
    x = np.arange(3)  # the label locations
    width = 0.10  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 10))
    
    for model in MODELS:
        df_model = df[df['Model'] == model]
        # order of datasets in column Datasets
        # df_model = df_model.reindex(datasets)
        # plt.plot(df_model.index, df_model[metric_mean], label=model, marker='o', markersize=10)
        # plt.errorbar(df_model.index, df_model[metric_mean], yerr=df_model['Double std'], capsize=4, fmt='none', color='black')
        offset = width * multiplier
        ax.bar(x + offset, df_model['RMSE mean'], width, label=model, yerr=df_model['Double std'], capsize=4)
        ax.plot(x + offset, df_model['RMSE mean'], marker='o', markersize=10, markeredgecolor='black')
        multiplier += 1
        
    plt.xticks(fontsize=20)
    plt.ylim((y_min, y_max))
    plt.title(f'Comparación de los datasets - {metric}', fontsize=25)
    plt.xlabel('')
    plt.yticks(fontsize=22)
    ax.set_xticks(x + width, datasets)
    ax.tick_params(axis='x', which='major', pad=15)
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    # plt.legend(fontsize=16)
    # plt.gca().legend_.remove()
    plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.25, 1.0))

    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_data_learning_folder_by_rule_type(rule_type)
    file = f'{test_figures_folder}/rule_type_comparison_{metric}_{rule_type}_{model_variation}_{num_individuals}ind{suffix}.png'
    if show:
        # tight
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(file, dpi=300, bbox_inches='tight')
        print(f'Figure saved in {file}')

