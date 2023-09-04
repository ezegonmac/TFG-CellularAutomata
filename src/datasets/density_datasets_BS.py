import numpy as np

from constants import *
from datasets.dataset_density_generation import generate_density_dataset_files_from_individuals
from datasets.CA_individual import CA_individual
from utils import *


def generate_dataset14_density(n_individuals) -> None:
    """
    Generate dataset14.
    
    Description:
    (Same as dataset3 and dataset11 but with rules B/S)
    Free thresholds, fixed density.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - free B: [bi,...] with bi in [0, 9] with same probability
    - free S: [si,...] with si in [0, 9] with same probability
    """
    
    dataset_name = DATASET14_DENSITY
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # free attributes
    B_min = 0
    B_max = 9
    S_min = 0
    S_max = 9
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # individuals with random B and S
    np.random.seed(NP_RANDOM_SEED)
    individuals = []
    for i in range(n_individuals):
        
        B_range = range(B_min, B_max)
        S_range = range(S_min, S_max)
        
        B_count = np.random.randint(B_min, B_max+1)
        S_count = np.random.randint(S_min, S_max+1)
        
        B = np.random.choice(B_range, B_count, replace=False)
        S = np.random.choice(S_range, S_count, replace=False)

        individuals.append(CA_individual(
            id=i,
            B=B, 
            S=S, 
            size=size, 
            density=density, 
            iterations=iterations, 
            file=f'{individuals_folder}/ca_{i}',
            ))
    
    generate_density_dataset_files_from_individuals(
        individuals, dataset_folder,
        rule_type='BS'
        )


def generate_dataset15_density(n_individuals) -> None:
    """
    Generate dataset15.

    Description:
    Same as dataset14 but with free density.
    
    Variables:
    - fixed size: 10x10
    - free density: [0, 1]
    - fixed iterations: 10
    - free B: [bi,...] with bi in [0, 9] with same probability
    - free S: [si,...] with si in [0, 9] with same probability
    """
    
    dataset_name = DATASET15_DENSITY
    
    # fixed attributes
    size = 10
    iterations = 10
    
    # free attributes
    B_min = 0
    B_max = 9
    S_min = 0
    S_max = 9
    density_min = 0
    density_max = 1
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # individuals with random B and S
    np.random.seed(NP_RANDOM_SEED)
    individuals = []
    for i in range(n_individuals):
        
        B_range = range(B_min, B_max)
        S_range = range(S_min, S_max)
        
        B_count = np.random.randint(B_min, B_max+1)
        S_count = np.random.randint(S_min, S_max+1)
        
        B = np.random.choice(B_range, B_count, replace=False)
        S = np.random.choice(S_range, S_count, replace=False)

        individuals.append(CA_individual(
            id=i,
            B=B, 
            S=S, 
            size=size, 
            density=np.random.uniform(density_min, density_max),
            iterations=iterations, 
            file=f'{individuals_folder}/ca_{i}',
            ))
    
    generate_density_dataset_files_from_individuals(
        individuals, dataset_folder,
        rule_type='BS'
        )


def generate_dataset16_density(n_individuals) -> None:
    """
    Generate dataset16.
    
    Description:
    (Same as dataset15 but with 30 iterations)
    
    Variables:
    - fixed size: 10x10
    - free density: [0, 1]
    - fixed iterations: 30
    - free B: [bi,...] with bi in [0, 9] with same probability
    - free S: [si,...] with si in [0, 9] with same probability
    """
    
    dataset_name = DATASET16_DENSITY
    
    # fixed attributes
    size = 10
    iterations = 30
    
    # free attributes
    B_min = 0
    B_max = 9
    S_min = 0
    S_max = 9
    density_min = 0
    density_max = 1
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # individuals with random B and S
    np.random.seed(NP_RANDOM_SEED)
    individuals = []
    for i in range(n_individuals):
        
        B_range = range(B_min, B_max)
        S_range = range(S_min, S_max)
        
        B_count = np.random.randint(B_min, B_max+1)
        S_count = np.random.randint(S_min, S_max+1)
        
        B = np.random.choice(B_range, B_count, replace=False)
        S = np.random.choice(S_range, S_count, replace=False)

        individuals.append(CA_individual(
            id=i,
            B=B, 
            S=S, 
            size=size, 
            density=np.random.uniform(density_min, density_max),
            iterations=iterations, 
            file=f'{individuals_folder}/ca_{i}',
            ))
    
    generate_density_dataset_files_from_individuals(
        individuals, dataset_folder,
        rule_type='BS'
        )
