import matplotlib.pyplot as plt
from constants import *
import os
import numpy as np
from matplotlib.ticker import AutoMinorLocator

STATES_FIGURES_FOLDER = f'{FIGURES_FOLDER}/statistics/states'

def create_state_plot(dataset, individual=0, iteration=0, show=False, title="State"):
    
    dataset_folder = f'{DATA_FOLDER}/{dataset}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # load specified state
    individual_file = os.listdir(individuals_folder)[individual]
    states_file = f'{individuals_folder}/{individual_file}'
    states = np.load(states_file)
    state = states[iteration]
    
    plot_state(state, title)

    plt.show() if show else plt.savefig(f'{STATES_FIGURES_FOLDER}/state_it_{iteration}_individual_{individual}_{dataset}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_state(state, title):
    ax = plt.subplot(111)
    
    plt.matshow(state, cmap='Greys', vmin=0, vmax=1, fignum=0)
    
    # TODO: remove magic number?
    ticks = range(0, 9+1)
    ax.set(yticks=ticks, xticks=ticks, title=title)
    # grid
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # remove minor ticks # not working
    for tick in ax.xaxis.get_major_ticks():
        # tick.tick1On = tick.tick2On = False
        tick.tick1line.set_visible(False)
    ax.grid(which='minor', color='white', linewidth=1)
