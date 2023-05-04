import numpy as np

from constants import *
from datasets.dataset_density_generation import generate_density_dataset_files_from_individuals
from datasets.CA_individual import CA_individual
from utils import *


def generate_dataset8_density() -> None:
    """
    Generate dataset8 density.
    
    Description:
    All cells dies or become alive in the next iteration, for new rules.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (0, 0), (9, 9)
    """
    
    dataset_name = DATASET8_DENSITY
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # define subsets with same attributes
    n_subset1 = 100
    n_subset2 = 100
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    
    # subsets
    np.random.seed(NP_RANDOM_SEED)
    subset1 = [CA_individual(
        id=id,
        B=0, 
        S=0, 
        size=size, 
        density=density, 
        iterations=iterations, 
        # file=f'{individuals_folder}/all_live_{id}',
        ) for id in range(n_subset1)]
    
    subset2 = [CA_individual(
        id=id,
        B=9, 
        S=9, 
        size=size, 
        density=density, 
        iterations=iterations, 
        # file=f'{individuals_folder}/all_die_{id}',
        ) for id in range(n_subset2)]

    subsets = [subset1, subset2]
    
    # individuals contains all subset individuals
    individuals = [individual for subset in subsets for individual in subset]
    
    generate_density_dataset_files_from_individuals(individuals, dataset_folder)


def generate_dataset3_density() -> None:
    """
    Generate dataset3.
        
    Some cells die some become alive.
    Free thresholds.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - free B : [0, 9]
    - free S: [0, 9]
    """
    
    # subsets with all attributes
    dataset_name = DATASET3_DENSITY
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # free attributes
    B_min = 0
    B_max = 9
    B_min = 0
    S_max = 9
      
    n_individuals = 1000
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=np.random.randint(B_min, B_max+1), 
        S=np.random.randint(B_min, S_max+1), 
        size=size, 
        density=density, 
        iterations=iterations, 
        # file=f'{individuals_folder}/ca_{id}',
        ) for id in range(n_individuals)]
    
    generate_density_dataset_files_from_individuals(individuals, dataset_folder)


def generate_dataset9_density(n_individuals=500) -> None:
    """
    Generate dataset9.
    
    Some cells die some become alive.
    Free thresholds.
    Free density.
    
    Variables:
    - fixed size: 10x10
    - fixed iterations: 5
    - free B : [0, 9]
    - free S: [0, 9]
    - free density: [0, 1]
    """
    
    # subsets with all attributes
    dataset_name = DATASET9_DENSITY
    
    # fixed attributes
    size = 10
    iterations = 10
    
    # free attributes
    B_min = 0
    B_max = 9
    B_min = 0
    S_max = 9
    density_min = 0
    density_max = 1
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=np.random.randint(B_min, B_max+1), 
        S=np.random.randint(B_min, S_max+1), 
        size=size, 
        density=np.random.uniform(density_min, density_max),
        iterations=iterations, 
        # file=f'{individuals_folder}/ca_{id}',
        ) for id in range(n_individuals)]
    
    generate_density_dataset_files_from_individuals(individuals, dataset_folder)


def generate_dataset10_density() -> None:
    """
    Generate dataset10.
    
    Description:
    (Same as dataset9 but with more iterations)
    Some cells die some become alive.
    Free thresholds.
    Free density.
    Increased number of iterations.
    
    Variables:
    - fixed size: 10x10
    - fixed iterations: 30
    - free B : [0, 9]
    - free S: [0, 9]
    - free density: [0, 1]
    """
    
    # subsets with all attributes
    dataset_name = DATASET10_DENSITY
    
    # fixed attributes
    size = 10
    iterations = 30
    
    # free attributes
    B_min = 0
    B_max = 9
    B_min = 0
    S_max = 9
    density_min = 0
    density_max = 1
    
    n_individuals = 500
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=np.random.randint(B_min, B_max+1), 
        S=np.random.randint(B_min, S_max+1), 
        size=size, 
        density=np.random.uniform(density_min, density_max),
        iterations=iterations, 
        # file=f'{individuals_folder}/ca_{id}',
        ) for id in range(n_individuals)]
    
    generate_density_dataset_files_from_individuals(individuals, dataset_folder)
