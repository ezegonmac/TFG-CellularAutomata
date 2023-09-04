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
    
    # BISI
    Bls = np.zeros(num_individuals, dtype=int) if rule_type == 'BISI' else None
    Bts = np.zeros(num_individuals, dtype=int) if rule_type == 'BISI' else None
    Sls = np.zeros(num_individuals, dtype=int) if rule_type == 'BISI' else None
    Sts = np.zeros(num_individuals, dtype=int) if rule_type == 'BISI' else None
    
    # BS
    B0 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    B1 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    B2 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    B3 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    B4 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    B5 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    B6 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    B7 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    B8 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S0 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S1 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S2 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S3 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S4 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S5 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S6 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S7 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    S8 = np.zeros(num_individuals, dtype=int) if rule_type == 'BS' else None
    
    
    densities = np.zeros((num_individuals, individuals[0].iterations))
    for i in range(num_individuals):
        individual = individuals[i]
        
        ca = generate_ca_from_individual(individual, rule_type, from_initial_state=False)
        
        density_evolution = np.array(ca.get_density_evolution())
        individual.density_evolution = density_evolution
        
        if i % 100 == 0:
            print(f'Individual {i}')
            print('...')
        
        # data = id|B|S|density_evolution
        ids[i] = int(i)
        Bs[i] = individual.B
        Ss[i] = individual.S
        
        # BISI
        if rule_type == 'BISI':
            Bls[i] = individual.B[0]
            Bts[i] = individual.B[1]
            Sls[i] = individual.S[0]
            Sts[i] = individual.S[1]
        densities[i] = density_evolution
        
        # BS
        if rule_type == 'BS':
            B0[i] = 0 in individual.B
            B1[i] = 1 in individual.B
            B2[i] = 2 in individual.B
            B3[i] = 3 in individual.B
            B4[i] = 4 in individual.B
            B5[i] = 5 in individual.B
            B6[i] = 6 in individual.B
            B7[i] = 7 in individual.B
            B8[i] = 8 in individual.B
            S0[i] = 0 in individual.S
            S1[i] = 1 in individual.S
            S2[i] = 2 in individual.S
            S3[i] = 3 in individual.S
            S4[i] = 4 in individual.S
            S5[i] = 5 in individual.S
            S6[i] = 6 in individual.S
            S7[i] = 7 in individual.S
            S8[i] = 8 in individual.S
    
    # data = id|B|S|0|1|2|3|...|9
    data = {
        'id': ids,
        'B': Bs,
        'S': Ss,
        }
    
    # BISI
    # data = id|Bl|Bt|Sl|St|0|1|2|3|...|9 if rule_type == 'BISI'
    data = {**data, **{'Bl': Bls, 'Bt': Bts, 'Sl': Sls, 'St': Sts}} if rule_type == 'BISI' else data
    
    # BS
    # data = id|B0|B1|B2|B3|B4|B5|B6|B7|B8|S0|S1|S2|S3|S4|S5|S6|S7|S8|0|1|2|3|...|9 if rule_type == 'BS'
    data = {**data, **{'B0': B0, 'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4, 'B5': B5, 'B6': B6, 'B7': B7, 'B8': B8, 'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5, 'S6': S6, 'S7': S7, 'S8': S8}} if rule_type == 'BS' else data
    
    # add density evolutions to data
    data = {**data, **{i: densities[:, i] for i in range(individuals[0].iterations)}}
    
    # save density evolutions to csv
    df = pd.DataFrame(data)
    df.to_csv(f'{dataset_folder}/density_dataset.csv')
