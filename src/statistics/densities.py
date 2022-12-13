from constants import *
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    # better quality figures

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
    # better quality figures