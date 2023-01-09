import matplotlib.pyplot as plt
from constants import FIGURES_FOLDER
import numpy as np
from matplotlib.ticker import AutoMinorLocator

BASIC_FIGURES_FOLDER = f'{FIGURES_FOLDER}/doc'


def generate_states_figure():
    state_alive = 0
    state_dead = 1

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    fig.subplots_adjust(wspace=0.5, hspace=2)
    
    # Black cell subplot
    ax[0].imshow([[state_alive]], cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Célula viva", fontsize=40)
    # ax[0].axis("off")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # White cell subplot
    ax[1].imshow([[state_dead]], cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Célula muerta", fontsize=40)
    # ax[1].axis("off")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.savefig(f'{BASIC_FIGURES_FOLDER}/states.png', dpi=200, bbox_inches='tight')
    # plt.show()


def generate_grid_figure():
    n = 6
    state = np.zeros((n, n))
    ax = plt.subplot(111)
    
    plt.matshow(state, cmap='Greys', vmin=0, vmax=1, fignum=0)
    
    ticks = ['0', '1', '2', '...', '...', 'n-1']
    
    # set x ticks and labels
    ax.set_xticks(range(0, n))
    ax.set_xticklabels(ticks)
    # set y ticks and labels
    ax.set_yticks(range(0, n))
    ax.set_yticklabels(ticks)
    # set x label on top left
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    # set x label
    ax.set_xlabel('i', fontsize=18)
    # rotate y label

    # set y label
    ax.set_ylabel('j', fontsize=18, rotation=0)

    # grid
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # remove minor ticks # not working
    for tick in ax.xaxis.get_major_ticks():
        # tick.tick1On = tick.tick2On = False
        tick.tick1line.set_visible(False)
    ax.grid(which='minor', color='black', linewidth=1)
    
    plt.savefig(f'{BASIC_FIGURES_FOLDER}/grid.png', dpi=200, bbox_inches='tight')
    # plt.show()


def generate_moore_figure():
    n = 6
    state = np.array(
        [[1.0, 1.0, 1.0, 1.0, 1.0], 
         [1.0, 0.3, 0.3, 0.3, 1.0], 
         [1.0, 0.3, 0.0, 0.3, 1.0], 
         [1.0, 0.3, 0.3, 0.3, 1.0], 
         [1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    ax = plt.subplot(111)
    
    plt.matshow(state, cmap='gnuplot2', vmin=0, vmax=1, fignum=0)

    ticks = ['' for i in range(0, n-1)]
    ax.set(xticks=range(0, n-1), yticks=range(0, n-1))
    ax.set(xticklabels=ticks, yticklabels=ticks)

    # grid
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # remove minor ticks # not working
    for tick in ax.xaxis.get_major_ticks():
        # tick.tick1On = tick.tick2On = False
        tick.tick1line.set_visible(False)
    ax.grid(which='minor', color='black', linewidth=1)
    
    plt.savefig(f'{BASIC_FIGURES_FOLDER}/moore.png', dpi=200, bbox_inches='tight')
    # plt.show()


def generate_interval_figure():
    l1 = -1
    t1 = 5
    
    l2 = 12
    t2 = 7

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    
    fig.subplots_adjust(wspace=0.5, hspace=2)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)

    ax.get_yaxis().set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_position('center')
    ax.set_xticks(range(0, 10+1))
    
    ax.plot([l1, t1], [0.25, 0.25], linestyle='-', marker='o', color='red', markerfacecolor='black', markeredgecolor='black', markersize=10)
    ax.plot([l2, t2], [0.25, 0.25], linestyle='-', marker='o', color='red', markerfacecolor='black', markeredgecolor='black', markersize=10)

    plt.savefig(f'{BASIC_FIGURES_FOLDER}/interval.png', dpi=200, bbox_inches='tight')
    plt.show()
