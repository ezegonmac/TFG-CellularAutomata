import matplotlib.pyplot as plt
import numpy as np

from constants import *
from learning.scoring import *
from utils import *
from learning.density_dataset import *
from learning.models import *


def generate_dataset3_individuals_plots():
    generate_individuals_real_vs_predicted_plot_not_scaled(
        DATASET3_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=50,
        suffix=''
        )


def generate_dataset8_individuals_plots():
    generate_individuals_real_vs_predicted_plot_not_scaled(
        DATASET8_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=10,
        suffix=''
        )


def generate_dataset11_individuals_plots():
    generate_individuals_real_vs_predicted_plot_scaled(
        DATASET11_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=1, 
        suffix=''
        )


def generate_dataset12_individuals_plots():
    generate_individuals_real_vs_predicted_plot_scaled(
        DATASET12_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=10, 
        suffix=''
        )


def generate_individuals_real_vs_predicted_plot_not_scaled(dataset, model_name, num_individuals, iterations=10, suffix=None, show=False):

    split = get_dataset_density_train_test_split(dataset, scaled=False)
    X, y, X_train, X_test, y_train, y_test = split
    
    model = load_model_from_file(dataset, model_name)

    print(f'Model: {model}')
    
    np.random.seed(NP_RANDOM_SEED)
    random_idx = np.random.choice(y_test.shape[0], num_individuals-1, replace=False)
    for i in random_idx:
        generate_individual_real_vs_predicted_plot_not_scaled(
            dataset, 
            model_name, 
            individual=i, 
            split=split,
            model=model,
            iterations=iterations,
            suffix=f'random_{i}'
            )


def generate_individual_real_vs_predicted_plot_not_scaled(dataset, model_name, individual, split, model, iterations=10, suffix=None, show=False):
    X, y, X_train, X_test, y_train, y_test = split

    individual_real = y_test.iloc[individual]
    individual_pred = model.predict(X_test.iloc[individual].values.reshape(1, -1))

    densities_real = individual_real.values
    densities_pred = individual_pred[0]

    colors = plt.cm.jet(np.linspace(0,1,8))
    plt.plot(densities_real, alpha=1, color=colors[2], marker='o', linestyle='--')
    plt.plot(densities_pred, alpha=1, color=colors[6], marker='.', linestyle=':')
    plt.legend(['Real', 'Pred'])
    plt.ylim((-0.1, 1.1))
    
    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_test_figures_folder(dataset)
    individuals_folder = f'{test_figures_folder}/individuals/{dataset}'
    create_folder_if_not_exists(individuals_folder)
    plt.show() if show else plt.savefig(f'{individuals_folder}/individual_rvsp_{dataset}_{model_name}_i{individual}_{suffix}.png', dpi=300)
    plt.close()


def generate_individuals_real_vs_predicted_plot_scaled(dataset, model_name, num_individuals, iterations=10, suffix=None, show=False):
    split = get_dataset_density_train_test_split(dataset, scaled=False)
    X, y, X_train, X_test, y_train, y_test = split
    
    model = load_model_from_file(dataset, model_name)

    print(f'Model: {model}')
    
    np.random.seed(NP_RANDOM_SEED)
    random_idx = np.random.choice(y_test.shape[0], num_individuals-1, replace=False)
    for i in random_idx:
        generate_individual_real_vs_predicted_plot_scaled(
            dataset, 
            model_name, 
            individual=i, 
            split=split,
            model=model,
            iterations=iterations,
            suffix=f'random_{i}'
            )


def generate_individual_real_vs_predicted_plot_scaled(dataset, model_name, individual, split, model, iterations=10, suffix=None, show=False):
    X, y, X_train, X_test, y_train, y_test = split
    _, yscaler = get_x_y_scalers(dataset)

    # predict
    individual_real = y_test.iloc[individual]
    individual_pred = model.predict(X_test.iloc[individual].values.reshape(1, -1))

    # inverse scale
    individual_real = individual_real
    individual_pred = yscaler.inverse_transform(individual_pred)

    # get densities
    densities_real = individual_real.values
    densities_pred = individual_pred[0]

    # plot
    colors = plt.cm.jet(np.linspace(0,1,8))
    plt.plot(densities_real, alpha=1, color=colors[2], marker='o', linestyle='--')
    plt.plot(densities_pred, alpha=1, color=colors[6], marker='.', linestyle=':')
    plt.legend(['Real', 'Pred'])
    plt.ylim((-0.1, 1.1))

    # file
    suffix = f'_{suffix}' if suffix else ''
    test_figures_folder = get_test_figures_folder(dataset)
    individuals_folder = f'{test_figures_folder}/individuals/{dataset}'
    create_folder_if_not_exists(individuals_folder)
    plt.show() if show else plt.savefig(f'{individuals_folder}/individual_rvsp_{dataset}_{model_name}_i{individual}_{suffix}.png', dpi=300)
    plt.close()
