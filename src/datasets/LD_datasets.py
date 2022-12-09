from CA.CAFactory import CAFactory
import os
import numpy as np
import pandas as pd
from shutil import rmtree

def generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder):
    # delete old dataset
    if os.path.exists(individuals_folder):
        rmtree(individuals_folder)
    # create new dataset folder
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(individuals_folder):
        os.mkdir(individuals_folder)
    
    # generate file and set evolution densities for each individual
    for individual in individuals:
        generate_file_from_individual(individual)
        evolution_densities = calc_evolution_densities_from_file(individual.file)
        individual.evolution_densities = evolution_densities
    
    df = pd.DataFrame(data=individuals)
    df.to_csv(f'{dataset_folder}/dataset.csv')

def generate_file_from_individual(individual):
    
    ca1 = CAFactory.create_CA_LB(
        life_threshold=individual.life_threshold,
        death_threshold=individual.death_threshold,
        size=individual.size,
        density=individual.density,
        iterations=individual.iterations)
    
    ca1.save_evolution(individual.file)

def calc_evolution_densities_from_file(file):
    evolution_densities = []
    
    evolution = np.load(file + '.npy')
    
    iterations = evolution.shape[0]
    
    for it in range(0, iterations):
        state = evolution[it]
        evolution_densities.append(np.count_nonzero(state) / state.size)
    
    return evolution_densities
