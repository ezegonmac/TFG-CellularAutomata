import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from constants import *
from utils import *


def create_state_plot(dataset, individual=0, iteration=0, show=False, title="State"):
    
    data_datasets_folder = get_data_datasets_folder(dataset)
    dataset_folder = f'{data_datasets_folder}/{dataset}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # load specified state
    individual_file = os.listdir(individuals_folder)[individual]
    states_file = f'{individuals_folder}/{individual_file}'
    states = np.load(states_file)
    state = states[iteration]
    
    plot_state(state, title)

    states_figures_folder = get_states_figures_folder(dataset)
    plt.show() if show else plt.savefig(f'{states_figures_folder}/state_it_{iteration}_individual_{individual}_{dataset}.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_state(state, title):
    ax = plt.subplot(111)
    
    plt.matshow(state, cmap='Greys', vmin=0, vmax=1, fignum=0)
    
    # TODO: remove magic number?
    iterations = state.shape[0]
    ticks = range(0, iterations)
    ax.set(yticks=ticks, xticks=ticks, title=title)
    # grid
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # remove minor ticks # not working
    for tick in ax.xaxis.get_major_ticks():
        # tick.tick1On = tick.tick2On = False
        tick.tick1line.set_visible(False)
    ax.grid(which='minor', color='white', linewidth=1)
