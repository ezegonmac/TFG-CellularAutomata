import os
from shutil import rmtree

import numpy as np
import pandas as pd

from utils import *
from datasets.dataset_generation import generate_ca_from_individual


def generate_density_dataset_files_from_individuals(individuals, dataset_folder, rule_type='BTST'):
    # delete old dataset
    if os.path.exists(dataset_folder):
        rmtree(dataset_folder)
    # create new dataset folder
    create_folder_if_not_exists(dataset_folder)
    
    # generate density evolutions for each individual
    num_individuals = len(individuals)
    ids = np.zeros(num_individuals, dtype=int)
    Bs = np.zeros(num_individuals, dtype=type(individuals[0].B))
    Ss = np.zeros(num_individuals, dtype=type(individuals[0].S))
    Bls = np.zeros(num_individuals, dtype=int) if rule_type == 'BISI' else None
    Bts = np.zeros(num_individuals, dtype=int) if rule_type == 'BISI' else None
    Sls = np.zeros(num_individuals, dtype=int) if rule_type == 'BISI' else None
    Sts = np.zeros(num_individuals, dtype=int) if rule_type == 'BISI' else None
    densities = np.zeros((num_individuals, individuals[0].iterations))
    for i in range(num_individuals):
        individual = individuals[i]
        
        ca = generate_ca_from_individual(individual, rule_type)
        
        density_evolution = np.array(ca.get_density_evolution())
        individual.density_evolution = density_evolution
        
        if i % 100 == 0:
            print(f'Individual {i}')
            print('...')
        
        # data = id|B|S|density_evolution
        ids[i] = int(i)
        Bs[i] = individual.B
        Ss[i] = individual.S
        Bls[i] = individual.B[0] if rule_type == 'BISI' else None
        Bts[i] = individual.B[1] if rule_type == 'BISI' else None
        Sls[i] = individual.S[0] if rule_type == 'BISI' else None
        Sts[i] = individual.S[1] if rule_type == 'BISI' else None
        densities[i] = density_evolution
    
    # data = id|B|S|0|1|2|3|...|9
    data = {
        'id': ids,
        'B': Bs,
        'S': Ss,
        }
    # data = id|Bl|Bt|Sl|St|0|1|2|3|...|9 if rule_type == 'BISI'
    data = {**data, **{'Bl': Bls, 'Bt': Bts, 'Sl': Sls, 'St': Sts}} if rule_type == 'BISI' else data
    
    # add density evolutions to data
    data = {**data, **{i: densities[:, i] for i in range(individuals[0].iterations)}}
    
    # save density evolutions to csv
    df = pd.DataFrame(data)
    df.to_csv(f'{dataset_folder}/density_dataset.csv')
