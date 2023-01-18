import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from constants import *
from utils import *


def create_random_states_plots(dataset, num_individuals=20, min_iteration=0, max_iteration=4, show=False, title=""):
    datasets_folder = get_data_datasets_folder(dataset)
    dataset_file = f'{datasets_folder}/{dataset}/dataset.csv'
    dataset_size = len(open(dataset_file).readlines())
    np.random.seed(NP_RANDOM_SEED)
    choices = np.random.choice(range(0, dataset_size-1), num_individuals, replace=False)
    for i in choices:
        create_states_plot(dataset, individual=i, min_iteration=0, max_iteration=4)


def create_states_plot(dataset, individual=0, min_iteration=0, max_iteration=4, show=False, title=""):
    fig, axs = plt.subplots(1, 
                            max_iteration+1, 
                            figsize=(3*(max_iteration+1), 5), 
                            sharex=True, 
                            sharey=True
                            )
    iterations = range(min_iteration, max_iteration+1)
    
    for ax, iteration in zip(axs, iterations):
        state = get_state(dataset, individual, iteration)
        title = f'It. {iteration+1}'
        plot_state(state, title, ax)
    
    states_figures_folder = get_states_figures_folder(dataset)
    folder = f'{states_figures_folder}/{dataset}'
    create_folder_if_not_exists(folder)
    plt.show() if show else plt.savefig(f'{folder}/{dataset}_states_individual_{individual}_it{min_iteration}_{max_iteration}.png', dpi=200, bbox_inches='tight')
    plt.close()


def create_state_plot(dataset, individual=0, iteration=0, show=False, title=""):
    state = get_state(dataset, individual, iteration)
    
    plot_state(state, title)

    states_figures_folder = get_states_figures_folder(dataset)
    folder = f'{states_figures_folder}/{dataset}'
    create_folder_if_not_exists(folder)
    plt.show() if show else plt.savefig(f'{folder}/{dataset}/state_it_{iteration}_individual_{individual}_{dataset}.png', dpi=200, bbox_inches='tight')
    plt.close()


def get_state(dataset, individual=0, iteration=0):
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_folder = f'{data_datasets_folder}/{dataset}'
    individuals_folder = f'{dataset_folder}/individuals'
    # check if individuals folder exists
    if not os.path.isdir(individuals_folder):
        raise Exception(f'Individuals folder "{individuals_folder}" does not exist. Probably you have not generated it yet.')
    
    # load specified state
    individual_file = os.listdir(individuals_folder)[individual]
    states_file = f'{individuals_folder}/{individual_file}'
    # check if state file exists
    if not os.path.isfile(states_file):
        raise Exception(f'State file "{states_file}" does not exist. Probably you have not generated it yet.')
    states = np.load(states_file)
    state = states[iteration]
    return state

def plot_state(state, title, ax=None):
    ax = plt.subplot(111) if ax is None else ax
    
    ax.matshow(state, cmap='Greys', vmin=0, vmax=1)
    
    iterations = state.shape[0]
    ticks = range(0, iterations)
    ax.set(yticks=ticks, xticks=ticks, title=title)
    # set x ticks on bottom
    ax.xaxis.set_ticks_position('bottom')
    # grid
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # remove minor ticks # not working
    for tick in ax.xaxis.get_major_ticks():
        # tick.tick1On = tick.tick2On = False
        tick.tick1line.set_visible(False)
    ax.grid(which='minor', color='white', linewidth=1)
