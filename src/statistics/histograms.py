from ast import literal_eval
from statistics.utils import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import *
from utils import *

# density histograms

def create_dataset_density_histograms_plot(dataset, show=False, title="Histogramas de densidad por iteración", limit=None, suffix=''):
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_file = f'{data_datasets_folder}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    
    # limit to firsts rows (for df)
    if limit is not None and len(df) > limit:
        df = df[:limit]
        
    density_evolutions = df['density_evolution'].apply(literal_eval).values
    density_evolutions = np.stack(density_evolutions, axis=0)

    densities_per_iteration = np.transpose(density_evolutions)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)
    # Big figure to set common labels
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('Densidad', labelpad=15)
    ax.set_ylabel('Frecuencia', labelpad=15)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    
    # One figure per iteration
    iterations = range(0, 9)
    for iteration, ax in zip(iterations, axes.flatten()):
        plot_density_histogram(ax, densities_per_iteration[iteration], iteration)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(w_pad=2, h_pad=3)
    
    density_figures_folder = get_density_figures_folder(dataset)
    suffix = f'_{suffix}' if suffix else ''
    plt.show() if show else plt.savefig(f'{density_figures_folder}/density_histograms_{dataset}{suffix}.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_density_histogram(ax, densities, iteration):
    title = f'It. {iteration}'
    ax.hist(densities, bins=20, range=[0, 1], density=True, color=COLOR_PRIMARY)  # , color='blue'
    ax.set(ylim=(0, 10), title=title)
    # title fontsize
    ax.title.set_size(14)
    # plt.grid(color='white', lw=0.5, axis='x', which='both')
    
    # ax.set(ylim=(0, 1.1), xlabel='Iteraciones', ylabel='Densidad', title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # minor ticks fontsize
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
