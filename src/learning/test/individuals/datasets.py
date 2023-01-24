from constants import *
from learning.test.individuals.plots import \
    generate_individuals_real_vs_predicted_plots
from utils import *


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


def generate_dataset9_individuals_plots():
    generate_individuals_real_vs_predicted_plots(
        DATASET9_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=20,
        scaled=False,
        suffix=''
        )


def generate_dataset10_individuals_plots():
    generate_individuals_real_vs_predicted_plots(
        DATASET10_DENSITY, 
        NEURAL_NETWORK, 
        num_individuals=20,
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
