from CA.CAFactory import CAFactory
import os
import numpy as np

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
    subsets = [
        ['all_live', 0, 9],
        ['all_die', 9, 0],
    ]
    
    number_of_seeds = 10
    
    for subset in subsets:
        subset_folder = subset[0]
        life_threshold = subset[1]
        death_threshold = subset[2]
        
        folder = f'{DATA_FOLDER}/dataset1/{subset_folder}'
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        for seed in range(0, number_of_seeds):
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

def generate_dataset2():
    """
    Some cells die some become alive.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 3
    - fixed life_threshold and death_threshold: (1, 8), (3, 6), (5, 5), (6, 3), (8, 1)
    """
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
        
        folder = f'{DATA_FOLDER}/dataset2/{subset_folder}'
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        for seed in range(0, number_of_seeds):
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
