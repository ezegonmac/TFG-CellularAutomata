import os
import sys
from shutil import rmtree

import pandas as pd

from CA.CAFactory import CAFactory
from utils import *


def generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=False, rule_type='BTST', from_initial_state=False):
    # delete old dataset
    if os.path.exists(dataset_folder):
        rmtree(dataset_folder)
    # create new dataset folder
    create_folder_if_not_exists(dataset_folder)
    create_folder_if_not_exists(individuals_folder)
    
    # generate file and set evolution densities for each individual
    for i in range(len(individuals)):
        individual = individuals[i]
        
        if i % 100 == 0:
            print(f'Individual {i}')
            print('...')
        
        ca = generate_ca_from_individual(individual, rule_type, from_initial_state)
        
        generate_individual_file(ca, individual.file) if save_individuals else None

        individual.density_evolution = list(ca.get_density_evolution())
    
    # converts individuals to dict to save attributes instead of objects
    individuals_dict = [individual.__dict__ for individual in individuals]
    df = pd.DataFrame(data=individuals_dict)
    df.to_csv(f'{dataset_folder}/dataset.csv')


def generate_individual_file(ca, file):
    ca.save_evolution(file)


def generate_ca_from_individual(individual, rule_type, from_initial_state):
    # from initial state or not
    function_name = f'generate_{rule_type}_ca_from_individual' if not from_initial_state else f'generate_{rule_type}_ca_from_individual_from_initial_state'
    generate_ca_from_individual_for_rule_type = getattr(
        sys.modules[__name__], 
        function_name
        )
    ca = generate_ca_from_individual_for_rule_type(individual)
    return ca


def generate_BTST_ca_from_individual(individual):
    ca = CAFactory.create_CA_BTST(
        B=individual.B,
        S=individual.S,
        size=individual.size,
        density=individual.density,
        iterations=individual.iterations)
    return ca


def generate_BISI_ca_from_individual(individual):
    ca = CAFactory.create_CA_BISI(
        B=individual.B,
        S=individual.S,
        size=individual.size,
        density=individual.density,
        iterations=individual.iterations)
    return ca


def generate_BS_ca_from_individual(individual):
    ca = CAFactory.create_CA_BS(
        B=individual.B,
        S=individual.S,
        size=individual.size,
        density=individual.density,
        iterations=individual.iterations)
    return ca


def generate_BTST_ca_from_individual_from_initial_state(individual):
    ca = CAFactory.create_CA_BTST(
        B=individual.B,
        S=individual.S,
        board=individual.initial_state,
        iterations=individual.iterations)
    return ca


def generate_BISI_ca_from_individual_from_initial_state(individual):
    ca = CAFactory.create_CA_BISI(
        B=individual.B,
        S=individual.S,
        board=individual.initial_state,
        iterations=individual.iterations)
    return ca


def generate_BS_ca_from_individual_from_initial_state(individual):
    ca = CAFactory.create_CA_BS(
        B=individual.B,
        S=individual.S,
        board=individual.initial_state,
        iterations=individual.iterations)
    return ca
