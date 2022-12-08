from CAFactory import CAFactory
import os
import numpy as np

DATA_FOLDER = './data'

def generate_dataset1():
    folder = f'{DATA_FOLDER}/dataset1'
    
    for seed in range(0, 10):
        # different seed for every iteration
        np.random.seed(seed)
            
        ca1 = CAFactory.create_CA_LB(
            life_threshold=2, 
            death_threshold=4, 
            size=10, 
            density=0.5, 
            iterations=3)
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file = f"{folder}/ca_s{seed}"
        ca1.save_evolution(file)