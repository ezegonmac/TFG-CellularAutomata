from ast import literal_eval
from statistics.utils import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from constants import *
from utils import *


def create_dataset_chaotic_evolution_plots(dataset, show=False, title="Evolución de la densidad", clusters=10, n_per_cluster=100, suffix='', num_colors=20):
    
    for cluster in range(1, clusters+1):
        create_dataset_chaotic_evolution_plot(dataset, show, title, n_per_cluster, cluster, suffix, num_colors)

def create_dataset_chaotic_evolution_plot(dataset, show=False, title="Evolución de la densidad", n_per_cluster=100, cluster=1, suffix='', num_colors=20):
    
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_file = f'{data_datasets_folder}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    
    low_limit = (cluster - 1) * n_per_cluster
    high_limit = cluster * n_per_cluster
    df = df.iloc[low_limit:high_limit]
    
    density_evolutions = df['density_evolution'].apply(literal_eval).values

    plot_chaotic_density_evolutions(density_evolutions, title, num_colors=num_colors)
    
    chaotic_figures_folder = get_chaotic_figures_folder(dataset)
    suffix = f'_{suffix}' if suffix else ''
    plt.show() if show else plt.savefig(f'{chaotic_figures_folder}/density_chaotic_{dataset}_c{cluster}{suffix}.png', dpi=300)
    plt.close()


def plot_chaotic_density_evolutions(density_evolutions, title, num_colors=20):
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
