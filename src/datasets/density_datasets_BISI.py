import numpy as np

from constants import *
from datasets.dataset_density_generation import generate_density_dataset_files_from_individuals
from datasets.CA_individual import CA_individual
from utils import *


def generate_dataset11_density(n_individuals) -> None:
    """
    Generate dataset11.
    
    Description:
    (Same as dataset3 but with rules BI/SI)
    Free thresholds, fixed density.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - free B: (bmin, bmax) with bmin, bmax in [0, 9]
    - free S: (smin, smax) with smin, smax in [0, 9]
    """
    
    dataset_name = DATASET11_DENSITY
    
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
        
        b1 = np.random.randint(B_min, B_max)
        b2 = np.random.randint(B_min, B_max)
        B = (b1, b2) if b1 < b2 else (b2, b1)
        
        s1 = np.random.randint(S_min, S_max)
        s2 = np.random.randint(S_min, S_max)
        S = (s1, s2) if s1 < s2 else (s2, s1)

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
        rule_type='BISI'
        )


def generate_dataset12_density(n_individuals) -> None:
    """
    Generate dataset12.
    
    Description:
    Some cells die some become alive.
    Free thresholds.
    Free density.
    
    Variables:
    - fixed size: 10x10
    - fixed iterations: 10
    - free B: (bmin, bmax) with bmin, bmax in [0, 9]
    - free S: (smin, smax) with smin, smax in [0, 9]
    - free density: [0, 1]
    """
    
    # subsets with all attributes
    dataset_name = DATASET12_DENSITY
    
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
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = []
    for i in range(n_individuals):
        
        b1 = np.random.randint(B_min, B_max)
        b2 = np.random.randint(B_min, B_max)
        B = (b1, b2) if b1 < b2 else (b2, b1)
        
        s1 = np.random.randint(S_min, S_max)
        s2 = np.random.randint(S_min, S_max)
        S = (s1, s2) if s1 < s2 else (s2, s1)

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
        rule_type='BISI'
        )


def generate_dataset13_density(n_individuals) -> None:
    """
    Generate dataset13.
    
    Description:
    (Same as dataset12 but with more iterations)
    Some cells die some become alive.
    Free thresholds.
    Free density.
    
    Variables:
    - fixed size: 10x10
    - fixed iterations: 50
    - free B: (bmin, bmax) with bmin, bmax in [0, 9]
    - free S: (smin, smax) with smin, smax in [0, 9]
    - free density: [0, 1]
    """
    
    # subsets with all attributes
    dataset_name = DATASET13_DENSITY
    
    # fixed attributes
    size = 10
    iterations = 50
    
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
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = []
    for i in range(n_individuals):
        
        b1 = np.random.randint(B_min, B_max)
        b2 = np.random.randint(B_min, B_max)
        B = (b1, b2) if b1 < b2 else (b2, b1)
        
        s1 = np.random.randint(S_min, S_max)
        s2 = np.random.randint(S_min, S_max)
        S = (s1, s2) if s1 < s2 else (s2, s1)

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
        rule_type='BISI'
        )
