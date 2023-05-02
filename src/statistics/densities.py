from ast import literal_eval
from statistics.utils import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from constants import *
from utils import *


# density evolution

def create_dataset_density_evolution_plot(dataset, show=False, title="Evolución de la densidad", start=0, limit=None, suffix='', num_colors=20):
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_file = f'{data_datasets_folder}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    
    # limit to firsts rows (for df)
    if limit is not None and len(df) > limit:
        df = df[start:start+limit]
    elif start > 0:
        df = df[start:]
    
    density_evolutions = df['density_evolution'].apply(literal_eval).values

    plot_density_evolutions(density_evolutions, title, num_colors=num_colors)
    
    density_figures_folder = get_density_figures_folder(dataset)
    suffix = f'_{suffix}' if suffix else ''
    plt.show() if show else plt.savefig(f'{density_figures_folder}/density_evolution_{dataset}{suffix}.png', dpi=250)
    plt.close()


def plot_density_evolutions(density_evolutions, title, num_colors=20):
    ax = plt.subplot(111)
    
    # fig size
    iterations = len(density_evolutions[0])
    if iterations <= 15:
        ax.figure.set_size_inches(6, 4)
    if iterations > 15:
        ax.figure.set_size_inches(15, 4)
    
    colors = plt.cm.jet(np.linspace(0,1,num_colors))
    for i in range(len(density_evolutions)):
        densities = density_evolutions[i]
        plt.plot(densities, alpha=0.5, color=colors[i % len(colors)])
    
    ax.set(ylim=(0, 1.1), xlabel='Iteraciones', ylabel='Densidad', title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# density variation

def create_dataset_density_variation_plot(dataset, show=False, title="Variación de la densidad", limit=None, suffix=''):
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_file = f'{data_datasets_folder}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    
    # limit to firsts rows (for df)
    if limit is not None and len(df) > limit:
        df = df[:limit]
        
    density_evolutions = df['density_evolution'].apply(literal_eval).values
    
    plot_density_variations(density_evolutions, title)
    
    density_figures_folder = get_density_figures_folder(dataset)
    suffix = f'_{suffix}' if suffix else ''
    plt.show() if show else plt.savefig(f'{density_figures_folder}/density_variation_{dataset}{suffix}.png', dpi=300)
    plt.close()
    

def plot_density_variations(density_evolutions, title):
    ax = plt.subplot(111)
    
    # fig size
    iterations = len(density_evolutions[0])
    if iterations <= 15:
        ax.figure.set_size_inches(6, 4)
    if iterations > 15:
        ax.figure.set_size_inches(15, 4)
    
    density_variations = []
    for density_evolution in density_evolutions:
        variations = []
        for i in range(len(density_evolution)-1):
            variation = density_evolution[i] - density_evolution[i-1]
            variations.append(variation)
        density_variations.append(variations)
    
    for variation in density_variations:
        plt.plot(variation, alpha=0.3)
    
    ax.set(ylim=(-1.1, 1.1), xlabel='Iteraciones', ylabel='Variación de la densidad', title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# density evolution per B

def create_dataset_density_evolution_per_B_plot(dataset, show=False, title="Evolución de la densidad por umbral B", limit=None, suffix=''):
    
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_file = f'{data_datasets_folder}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    
    # limit to firsts rows (for df)
    if limit is not None and len(df) > limit:
        df = df[:limit]
    
    Bs = df['B'].unique().tolist()
    
    density_evolutions_by_B = {}
    for B in Bs:
        density_evolutions = df[df['B'] == B]['density_evolution'].apply(literal_eval).values
        density_evolutions_by_B[B] = density_evolutions
    
    title = title+', B=0' if dataset == DATASET4 else title
    title = title+', S=0' if dataset == DATASET5 else title
    title = title+', B=9' if dataset == DATASET6 else title
    title = title+', S=9' if dataset == DATASET7 else title
    plot_density_evolutions_per_B(density_evolutions_by_B, Bs, title)
    
    density_figures_folder = get_density_figures_folder(dataset)
    suffix = f'_{suffix}' if suffix else ''
    plt.show() if show else plt.savefig(f'{density_figures_folder}/density_evolution_B_{dataset}{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_density_evolutions_per_B(density_evolutions_by_B, Bs, title):
    ax = plt.subplot(111)
    
    colors_by_B = get_colors_by_threshold()
    for B in Bs:
        density_evolutions = density_evolutions_by_B[B]
        color = colors_by_B[B]
        for densities in density_evolutions:
            plt.plot(densities, c=color, alpha=0.3)
    
    ax.set(ylim=(0, 1.1), xlabel='Iteraciones', ylabel='Densidad', title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # legend
    custom_lines = [Line2D([0], [0], color=color, lw=4)
                    for color in colors_by_B.values()]
    ax.legend(title="B",fancybox=True,
              handles=custom_lines, 
              labels=range(0, 10),
              loc='center left', bbox_to_anchor=(1, 0.5))

# density evolution per S

def create_dataset_density_evolution_per_S_plot(dataset, show=False, title="Evolución de la densidad por umbral S", limit=None, suffix=''):
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_file = f'{data_datasets_folder}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    
    # limit to firsts rows (for df)
    if limit is not None and len(df) > limit:
        df = df[:limit]
    
    Ss = df['S'].unique().tolist()
        
    density_evolutions_by_S = {}
    for S in Ss:
        density_evolutions = df[df['S'] == S]['density_evolution'].apply(literal_eval).values
        density_evolutions_by_S[S] = density_evolutions
    
    title = title+', B=0' if dataset == DATASET4 else title
    title = title+', S=0' if dataset == DATASET5 else title
    title = title+', B=9' if dataset == DATASET6 else title
    title = title+', S=9' if dataset == DATASET7 else title
    plot_density_evolutions_per_S(density_evolutions_by_S, Ss, title)
    
    density_figures_folder = get_density_figures_folder(dataset)
    suffix = f'_{suffix}' if suffix else ''
    plt.show() if show else plt.savefig(f'{density_figures_folder}/density_evolution_S_{dataset}{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_density_evolutions_per_S(density_evolutions_by_S, Ss, title):
    ax = plt.subplot(111)
    
    colors_by_threshold = get_colors_by_threshold()
    for S in Ss:
        density_evolutions = density_evolutions_by_S[S]
        color = colors_by_threshold[S]
        for densities in density_evolutions:
            plt.plot(densities, c=color, alpha=0.3)
    
    ax.set(ylim=(0, 1.1), xlabel='Iteraciones', ylabel='Densidad', title=title)
    # set label fontsize
    ax.axes.xaxis.label.set_size(12)
    ax.axes.yaxis.label.set_size(12)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # legend
    custom_lines = [Line2D([0], [0], color=color, lw=4)
                    for color in colors_by_threshold.values()]
    ax.legend(title="S",fancybox=True,
              handles=custom_lines, 
              labels=range(0, 10),
              loc='center left', bbox_to_anchor=(1, 0.5))

# density evolution per B and S

def plot_density_evolutions_per_B_and_S(density_evolutions_by_B_and_S, BSs, title):
    ax = plt.subplot(111)
    
    colors_by_threshold = get_colors_by_threshold()
    for BS in BSs:
        B = BS[0]
        S = BS[1]
        density_evolutions = density_evolutions_by_B_and_S[BS]
        color = colors_by_threshold[B]
        for densities in density_evolutions:
            plt.plot(densities, c=color, alpha=0.3)
    
    ax.set(ylim=(0, 1.1), xlabel='Iteraciones', ylabel='Densidad', title=title)
    # set label fontsize
    ax.axes.xaxis.label.set_size(12)
    ax.axes.yaxis.label.set_size(12)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # legend
    custom_lines = [Line2D([0], [0], color=color, lw=4)
                    for color in colors_by_threshold.values()]
    # only first and last
    custom_lines = [custom_lines[0], custom_lines[-1]]
    ax.legend(title="(B, S)",fancybox=True,
              handles=custom_lines, 
              labels=[(0, 0), (9, 9)],
              loc='center left', bbox_to_anchor=(0.9, 0.5))

def create_dataset_density_evolution_per_B_and_S_plot(dataset, show=False, title="Evolución de la densidad por umbrales B y S", limit=None, suffix=''):
    # Used only for equal values of B and S
    # Sometimes only takes B in account
    
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_file = f'{data_datasets_folder}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    
    # limit to firsts rows (for df)
    if limit is not None and len(df) > limit:
        df = df[:limit]
    
    Bs = df['B'].unique().tolist()
    Ss = df['S'].unique().tolist()
    BSs = list(zip(Bs, Ss))
    
    density_evolutions_by_B_and_S = {}
    for B, S in BSs:
        # Only takes b in account
        density_evolutions = df[df['B'] == B]['density_evolution'].apply(literal_eval).values
        density_evolutions_by_B_and_S[(B,S)] = density_evolutions
    
    plot_density_evolutions_per_B_and_S(density_evolutions_by_B_and_S, BSs, title)
    
    density_figures_folder = get_density_figures_folder(dataset)
    suffix = f'_{suffix}' if suffix else ''
    plt.show() if show else plt.savefig(f'{density_figures_folder}/density_evolution_B_and_S_{dataset}{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
