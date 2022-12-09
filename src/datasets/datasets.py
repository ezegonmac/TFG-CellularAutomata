from CA.CAFactory import CAFactory
import os
import numpy as np
import pandas as pd

DATA_FOLDER = './data'

def generate_dataset1():
    """
    All cells dies or become alive in the next iteration.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 3
    - fixed life_threshold and death_threshold: (0, 9), (9, 0)
    """
    dataset_name = 'dataset1'
    
    subsets = [
        ['all_live', 0, 9],
        ['all_die', 9, 0],
    ]
    
    number_of_seeds = 10
    
    seeds = []
    life_thresholds = []
    death_thresholds = []
    files = []
    for subset in subsets:
        subset_folder = subset[0]
        life_threshold = subset[1]
        death_threshold = subset[2]
        subset_seeds = range(0, number_of_seeds)
        
        subset_files = save_files_LD(dataset_name, subset_seeds, subset_folder, life_threshold, death_threshold)

        # add subset attributes to lists
        seeds.extend(subset_seeds)
        life_thresholds.extend([life_threshold] * number_of_seeds)
        death_thresholds.extend([death_threshold] * number_of_seeds)
        files.extend(subset_files)

    df = pd.DataFrame(
        data={'seed' : seeds,
              'life_threshold' : life_thresholds, 
              'death_threshold' : death_thresholds, 
              'file' : files,
              })
    
    df.to_csv(f'{DATA_FOLDER}/{dataset_name}/dataset.csv')

def save_files_LD(dataset_name, seeds, subset_folder, life_threshold, death_threshold):
    folder = f'{DATA_FOLDER}/{dataset_name}/{subset_folder}'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    subset_files = []
    for seed in seeds:
        # different seed for every iteration
        np.random.seed(seed)
            
        ca1 = CAFactory.create_CA_LB(
            life_threshold=life_threshold,
            death_threshold=death_threshold,
            size=10,
            density=0.5,
            iterations=3)
        
        file = f"{folder}/ca_s{seed}"
        ca1.save_evolution(file)
        subset_files.append(file)
        
    return subset_files

# not working
def generate_dataset2():
    """
    Some cells die some become alive.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 3
    - fixed life_threshold and death_threshold: (1, 8), (3, 6), (5, 5), (6, 3), (8, 1)
    """
    dataset_name = 'dataset2'
    
    subsets = [
        ['l1_d8', 1, 8],
        ['l3_d6', 3, 6],
        ['l5_d5', 5, 5],
        ['l6_d3', 6, 3],
        ['l8_d1', 8, 1],
    ]
    
    number_of_seeds = 4
    
    for subset in subsets:
        subset_folder = subset[0]
        life_threshold = subset[1]
        death_threshold = subset[2]
        
        save_file_LD(dataset_name, number_of_seeds, subset_folder, life_threshold, death_threshold)

