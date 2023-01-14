import numpy as np

from constants import *
from datasets.dataset_generation import generate_dataset_files_from_individuals
from datasets.CA_individual import CA_individual
from utils import *


def generate_dataset11(save_individuals=False) -> None:
    """
    Generate dataset11.
    
    Description:
    (Same as dataset8 but with rules BI/SI)
    All cells dies or become alive in the next iteration.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (0, 0), (9, 9)
    """
    
    dataset_name = DATASET11
    
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
    individuals_folder = f'{dataset_folder}/individuals'
    
    # subsets
    np.random.seed(NP_RANDOM_SEED)
    subset1 = [CA_individual(
        id=id,
        B=0, 
        S=0, 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/all_live_{id}',
        ) for id in range(n_subset1)]
    
    subset2 = [CA_individual(
        id=id,
        B=9, 
        S=9, 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/all_die_{id}',
        ) for id in range(n_subset2)]

    subsets = [subset1, subset2]
    
    # individuals contains all subset individuals
    individuals = [individual for subset in subsets for individual in subset]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals)
