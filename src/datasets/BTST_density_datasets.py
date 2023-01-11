from CA.CAFactory import CAFactory
import os
import numpy as np
import pandas as pd
from shutil import rmtree

def generate_density_dataset_files_from_individuals(individuals, dataset_folder):
    # delete old dataset
    if os.path.exists(dataset_folder):
        rmtree(dataset_folder)
    # create new dataset folder
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    
    # generate density evolutions for each individual
    num_individuals = len(individuals)
    ids = np.zeros(num_individuals, dtype=int)
    Bs = np.zeros(num_individuals, dtype=int)
    Ss = np.zeros(num_individuals, dtype=int)
    densities = np.zeros((num_individuals, individuals[0].iterations))
    for individual in individuals:
        density_evolution = generate_density_evolution_from_individual(individual)
        density_evolution = np.array(density_evolution)
        id = int(individual.id)
        
        # data = id|B|S|density_evolution
        ids[id] = int(individual.id)
        Bs[id] = individual.B
        Ss[id] = individual.S
        densities[id] = density_evolution
    
    # data = id|B|S|0|1|2|3|...|9
    data = {
            'id': ids,
            'B': Bs,
            'S': Ss,
        }
    data = {**data, **{i: densities[:, i] for i in range(individuals[0].iterations)}}
    
    # save density evolutions to csv
    df = pd.DataFrame(data)
    df.to_csv(f'{dataset_folder}/density_dataset.csv')

def generate_density_evolution_from_individual(individual):
    
    ca1 = CAFactory.create_CA_BTST(
        B=individual.B,
        S=individual.S,
        size=individual.size,
        density=individual.density,
        iterations=individual.iterations)
    
    return ca1.get_density_evolution()
