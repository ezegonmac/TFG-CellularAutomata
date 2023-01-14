import os
from shutil import rmtree

import numpy as np
import pandas as pd

from CA.CAFactory import CAFactory
from utils import *


def generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder):
    # delete old dataset
    if os.path.exists(dataset_folder):
        rmtree(dataset_folder)
    # create new dataset folder
    create_folder_if_not_exists(dataset_folder)
    create_folder_if_not_exists(individuals_folder)
    
    # generate file and set evolution densities for each individual
    for individual in individuals:
        generate_file_from_individual(individual)
        density_evolution = calc_density_evolution_from_file(individual.file)
        individual.density_evolution = density_evolution
    
    # converts individuals to dict to save attributes instead of objects
    individuals_dict = [individual.__dict__ for individual in individuals]
    df = pd.DataFrame(data=individuals_dict)
    df.to_csv(f'{dataset_folder}/dataset.csv')

def generate_file_from_individual(individual):
    
    ca1 = CAFactory.create_CA_BTST(
        B=individual.B,
        S=individual.S,
        size=individual.size,
        density=individual.density,
        iterations=individual.iterations)
    
    ca1.save_evolution(individual.file)

def calc_density_evolution_from_file(file):
    density_evolution = []
    
    evolution = np.load(file + '.npy')

    iterations = evolution.shape[0]
    
    for it in range(0, iterations):
        state = evolution[it]
        density = np.count_nonzero(state) / state.size
        density_evolution.append(density)
    
    return density_evolution
