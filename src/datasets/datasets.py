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
        ['all_die', 9, 0]
    ]
    
    for subset in subsets:
        subset_folder = subset[0]
        life_threshold = subset[1]
        death_threshold = subset[2]
        
        folder = f'{DATA_FOLDER}/dataset1/{subset_folder}'
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        for seed in range(0, 10):
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
