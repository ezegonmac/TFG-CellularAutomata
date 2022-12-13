from constants import *
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import numpy as np

DENSITIES_FIGURES_FOLDER = f'{FIGURES_FOLDER}/statistics/densities'

# density evolution

def create_dataset_density_evolution_plot(dataset, show=False, title="Density evolution"):
    
    dataset_file = f'{DATA_FOLDER}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    density_evolutions = df['density_evolution'].apply(literal_eval).values
    
    plot_density_evolutions(density_evolutions, title)
    
    plt.show() if show else plt.savefig(f'{DENSITIES_FIGURES_FOLDER}/density_evolution_{dataset}.png', dpi=300)
    plt.close()
    

def plot_density_evolutions(density_evolutions, title):
    ax = plt.subplot(111)
    
    for densities in density_evolutions:
        plt.plot(densities, alpha=0.5)
    
    ax.set(ylim=(0, 1.1), xlabel='Iterations', ylabel='Density', title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# density variation

def create_dataset_density_variation_plot(dataset, show=False, title="Density variations"):
    
    dataset_file = f'{DATA_FOLDER}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    density_evolutions = df['density_evolution'].apply(literal_eval).values
    
    plot_density_variations(density_evolutions, title)
    
    plt.show() if show else plt.savefig(f'{DENSITIES_FIGURES_FOLDER}/density_variation_{dataset}.png', dpi=300)
    plt.close()
    

def plot_density_variations(density_evolutions, title):
    
    density_variations = []
    for density_evolution in density_evolutions:
        variations = []
        for i in range(len(density_evolution)-1):
            variation = density_evolution[i] - density_evolution[i-1]
            variations.append(variation)
        density_variations.append(variations)
    
    ax = plt.subplot(111)
    
    for variation in density_variations:
        plt.plot(variation, alpha=0.5)
    
    ax.set(ylim=(-1.1, 1.1), xlabel='Iterations', ylabel='Density variation', title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# density evolution per life threshold

def create_dataset_density_evolution_per_lt_plot(dataset, show=False, title="Density evolution per life threshold"):
    
    dataset_file = f'{DATA_FOLDER}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    lts = df['life_threshold'].unique().tolist()
    
    density_evolutions_by_lt = {}
    for lt in lts:
        density_evolutions = df[df['life_threshold'] == lt]['density_evolution'].apply(literal_eval).values
        density_evolutions_by_lt[lt] = density_evolutions
    
    plot_density_evolutions_per_lt(density_evolutions_by_lt, lts, title)
    
    plt.show() if show else plt.savefig(f'{DENSITIES_FIGURES_FOLDER}/density_evolution_lt_{dataset}.png', dpi=300)
    plt.close()
    

def plot_density_evolutions_per_lt(density_evolutions_by_lt, lts, title):
    ax = plt.subplot(111)
    
    cmap = cm.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(lts)))
    colors_by_lt = {lts[i]: colors[i] for i in range(len(lts))}
    for lt in lts:
        density_evolutions = density_evolutions_by_lt[lt]
        color = colors_by_lt[lt]
        for densities in density_evolutions:
            plt.plot(densities, c=color, alpha=0.5)
    
    ax.set(ylim=(0, 1.1), xlabel='Iterations', ylabel='Density', title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# density evolution per life threshold

def create_dataset_density_evolution_per_dt_plot(dataset, show=False, title="Density evolution per life threshold"):
    
    dataset_file = f'{DATA_FOLDER}/{dataset}/dataset.csv'
    df = pd.read_csv(dataset_file)
    dts = df['death_threshold'].unique().tolist()
    
    density_evolutions_by_dt = {}
    for dt in dts:
        density_evolutions = df[df['death_threshold'] == dt]['density_evolution'].apply(literal_eval).values
        density_evolutions_by_dt[dt] = density_evolutions
    
    plot_density_evolutions_per_dt(density_evolutions_by_dt, dts, title)
    
    plt.show() if show else plt.savefig(f'{DENSITIES_FIGURES_FOLDER}/density_evolution_dt_{dataset}.png', dpi=300)
    plt.close()
    

def plot_density_evolutions_per_dt(density_evolutions_by_dt, dts, title):
    ax = plt.subplot(111)
    
    cmap = cm.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(dts)))
    colors_by_dt = {dts[i]: colors[i] for i in range(len(dts))}
    for dt in dts:
        density_evolutions = density_evolutions_by_dt[dt]
        color = colors_by_dt[dt]
        for densities in density_evolutions:
            plt.plot(densities, c=color, alpha=0.5)
    
    ax.set(ylim=(0, 1.1), xlabel='Iterations', ylabel='Density', title=title)
    # remove right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # integer x axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

