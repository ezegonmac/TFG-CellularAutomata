from datasets.CA_BTST_individual import CA_BTST_individual
from datasets.BTST_density_datasets import generate_density_dataset_files_from_individuals
import numpy as np
from constants import *


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
    
    dataset_name = DATASET8 + '_density'
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # define subsets with same attributes
    n_subset1 = 200
    n_subset2 = 200
    
    # folders
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    
    # subsets
    np.random.seed(NP_RANDOM_SEED)
    subset1 = [CA_BTST_individual(
        id=id,
        B=0, 
        S=0, 
        size=size, 
        density=density, 
        iterations=iterations, 
        # file=f'{individuals_folder}/all_live_{id}',
        ) for id in range(n_subset1)]
    
    subset2 = [CA_BTST_individual(
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
    - fixed iterations: 5
    - free B : [0, 9]
    - free S: [0, 9]
    """
    
    # subsets with all attributes
    dataset_name = DATASET3 + '_density'
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # free attributes
    lt_min = 0
    lt_max = 9
    dt_min = 0
    dt_max = 9
      
    n_individuals = 500
    
    # folders
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_BTST_individual(
        id=id,
        B=np.random.randint(lt_min, lt_max+1), 
        S=np.random.randint(dt_min, dt_max+1), 
        size=size, 
        density=density, 
        iterations=iterations, 
        # file=f'{individuals_folder}/ca_{id}',
        ) for id in range(n_individuals)]
    
    generate_density_dataset_files_from_individuals(individuals, dataset_folder)
