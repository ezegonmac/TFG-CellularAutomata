import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from constants import *
from learning.scoring import *
from utils import *
from learning.density_dataset import *
from learning.models import *


def generate_dataset3_individuals_plots():
    generate_individuals_real_vs_predicted_plots(
        DATASET3_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=20,
        scaled=False,
        suffix=''
        )


def generate_dataset8_individuals_plots():
    generate_individuals_real_vs_predicted_plots(
        DATASET8_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=10,
        scaled=False,
        suffix=''
        )


def generate_dataset11_individuals_plots():
    generate_individuals_real_vs_predicted_plots(
        DATASET11_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=20, 
        scaled=True,
        suffix=''
        )


def generate_dataset12_individuals_plots():
    generate_individuals_real_vs_predicted_plots(
        DATASET12_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=20, 
        scaled=True,
        suffix=''
        )


def generate_dataset13_individuals_plots():
    generate_individuals_real_vs_predicted_plots(
        DATASET13_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=20, 
        scaled=True,
        suffix=''
        )


def generate_individuals_real_vs_predicted_plots(dataset, model_name, num_individuals, iterations=10, scaled=False, suffix=None, show=False):
    split = get_dataset_density_train_test_split(dataset, scaled=scaled)
    X, y, X_train, X_test, y_train, y_test = split
    
    model = load_model_from_file(dataset, model_name)
    print(f'Model: {model}')
    
    np.random.seed(NP_RANDOM_SEED)
    random_idx = np.random.choice(y_test.shape[0], num_individuals, replace=False)
    for i in random_idx:
        generate_individual_real_vs_predicted_plot(
            dataset, 
            model_name, 
            individual=i, 
            split=split,
            model=model,
            iterations=iterations,
            scaled=scaled,
            suffix=f'random_{i}'
            )


def generate_individual_real_vs_predicted_plot(dataset, model_name, individual, split, model, iterations=10, scaled=False, suffix=None, show=False):
    X, y, X_train, X_test, y_train, y_test = split
    _, yscaler = get_x_y_scalers(dataset) if scaled else (None, None)

    # predict
    individual_real = np.array(y_test.iloc[individual]).reshape(1, -1)
    individual_pred = model.predict(X_test.iloc[individual].values.reshape(1, -1))

    # inverse scale
    individual_real = yscaler.inverse_transform(individual_real) if scaled else individual_real
    individual_pred = yscaler.inverse_transform(individual_pred) if scaled else individual_pred

    # get densities np arrays
    densities_real = individual_real[0]
    densities_pred = individual_pred[0]

    # plot
    # size and x ticks
    iterations = len(densities_real)
    if iterations <= 15:
        plt.figure(figsize=(6, 4))
        plt.xticks(np.arange(iterations), np.arange(1, iterations+1))
    if iterations > 15:
        plt.figure(figsize=(12, 4))
        plt.xticks(np.arange(0, iterations, 5), np.arange(1, iterations+1, 5))
    plt.ylim((-0.199, 1.199))
    plt.xlim((-0.5, iterations))
    # limits
    plt.plot([-1, iterations+0.5], [0, 0], color='black', linewidth=0.35, alpha=0.8, label='_nolegend_')
    plt.plot([-1, iterations+0.5], [1, 1], color='black', linewidth=0.35, alpha=0.8, label='_nolegend_')
    # remove spines top and right
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # labels
    plt.xlabel('Iteración', labelpad=10, fontsize=12)
    plt.ylabel('Densidad', labelpad=10, fontsize=12)
    
    colors = plt.cm.jet(np.linspace(0,1,8))
    plt.plot(densities_real, label='Real', alpha=1, color=colors[2], marker='o', linestyle='-')
    plt.plot(densities_pred, label='Predicho', alpha=1, color=colors[6], marker='.', linestyle=':')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))

    # file
    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_test_figures_folder(dataset)
    individuals_folder = f'{test_figures_folder}/individuals/{dataset}'
    create_folder_if_not_exists(individuals_folder)
    file = f'{individuals_folder}/individual_rvsp_{dataset}_{model_name}_i{individual}_{suffix}.png'
    plt.show() if show else plt.savefig(file, dpi=300, bbox_inches='tight')
    print(file)
    plt.close()
