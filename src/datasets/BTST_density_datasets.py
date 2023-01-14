import os
from shutil import rmtree

import numpy as np
import pandas as pd

from utils import *
from datasets.BTST_datasets import generate_BTST_ca_from_individual


def generate_density_dataset_files_from_individuals(individuals, dataset_folder):
    # delete old dataset
    if os.path.exists(dataset_folder):
        rmtree(dataset_folder)
    # create new dataset folder
    create_folder_if_not_exists(dataset_folder)
    
    # generate density evolutions for each individual
    num_individuals = len(individuals)
    ids = np.zeros(num_individuals, dtype=int)
    Bs = np.zeros(num_individuals, dtype=int)
    Ss = np.zeros(num_individuals, dtype=int)
    densities = np.zeros((num_individuals, individuals[0].iterations))
    for i in range(num_individuals):
        individual = individuals[i]
        
        ca = generate_BTST_ca_from_individual(individual)
        
        density_evolution = np.array(ca.get_density_evolution())
        individual.density_evolution = density_evolution
        
        # data = id|B|S|density_evolution
        ids[i] = int(i)
        Bs[i] = individual.B
        Ss[i] = individual.S
        densities[i] = density_evolution
    
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
