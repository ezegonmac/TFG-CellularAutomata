from datasets.LD_datasets import generate_dataset_files_from_individuals
import numpy as np
from constants import *

class CA_LD_individual:
    def __init__(self, id, life_threshold, death_threshold, size, density, iterations, file):
        self.id = id
        self.life_threshold = life_threshold
        self.death_threshold = death_threshold
        self.size = size
        self.density = density
        self.iterations = iterations
        self.file = file
        self.density_evolution = None

def generate_dataset1() -> None:
    """
    Generate dataset1.
    
    Description:
    All cells dies or become alive in the next iteration.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed life_threshold and death_threshold: (0, 9), (9, 0)
    """
    
    dataset_name = DATASET1
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # define subsets with same attributes
    n_subset1 = 20
    n_subset2 = 20
    
    # folders
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # subsets
    np.random.seed(NP_RANDOM_SEED)
    subset1 = [CA_LD_individual(
        id=id,
        life_threshold=0, 
        death_threshold=9, 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/all_live_{id}',
        ) for id in range(n_subset1)]
    
    subset2 = [CA_LD_individual(
        id=id,
        life_threshold=9, 
        death_threshold=0, 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/all_die_{id}',
        ) for id in range(n_subset2)]

    subsets = [subset1, subset2]
    
    # individuals contains all subset individuals
    individuals = [individual for subset in subsets for individual in subset]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder)

def generate_dataset2() -> None:
    """
    Generate dataset2.
    
    Some cells die some become alive.
    Fixed rules pairs.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed life_threshold and death_threshold: (1, 8), (3, 6), (5, 5), (6, 3), (8, 1)
    """
    
    dataset_name = DATASET2
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # folders
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    lt_dt_pairs = [(1, 8), (3, 6), (5, 5), (6, 3), (8, 1)]
    n_subset = 30
    
    # individuals with fixed lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_LD_individual(
        id=id,
        life_threshold=lt_dt_pairs[pair][0], 
        death_threshold=lt_dt_pairs[pair][1], 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/ca_pair_{pair}_{id}',
        ) for pair in range(len(lt_dt_pairs)) for id in range(n_subset)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder)

def generate_dataset3() -> None:
    """
    Generate dataset3.
        
    Some cells die some become alive.
    Free thresholds.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 5
    - free life_threshold : [0, 9]
    - free death_threshold: [0, 9]
    """
    
    # subsets with all attributes
    dataset_name = DATASET3
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # free attributes
    lt_min = 0
    lt_max = 9
    dt_min = 0
    dt_max = 9
      
    n_individuals = 300
    
    # folders
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_LD_individual(
        id=id,
        life_threshold=np.random.randint(lt_min, lt_max+1), 
        death_threshold=np.random.randint(dt_min, dt_max+1), 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/ca_{id}',
        ) for id in range(n_individuals)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder)

def generate_dataset4() -> None:
    """
    Generate dataset4.
    
    Some cells die some become alive.
    Fixed life threshold.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed life_threshold and death_threshold: (0, 0), (0, 1), ... (0, 9)
    """
    
    dataset_name = DATASET4
    
    generate_fixed_lt_dataset(dataset_name, 0)
    

def generate_dataset5() -> None:
    """
    Generate dataset5.
    
    Some cells die some become alive.
    Fixed death threshold.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed life_threshold and death_threshold: (0, 0), (1, 0), ... (9, 0)
    """
    
    dataset_name = DATASET5
    
    generate_fixed_dt_dataset(dataset_name, 0)


def generate_dataset6() -> None:
    """
    Generate dataset6.
    
    Some cells die some become alive.
    Fixed life threshold.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed life_threshold and death_threshold: (9, 0), (9, 1), ... (9, 9)
    """
    
    dataset_name = DATASET6
    
    generate_fixed_lt_dataset(dataset_name, 9)
    

def generate_dataset7() -> None:
    """
    Generate dataset7.
    
    Some cells die some become alive.
    Fixed death threshold.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed life_threshold and death_threshold: (0, 9), (1, 9), ... (9, 9)
    """
    
    dataset_name = DATASET7
    
    generate_fixed_dt_dataset(dataset_name, 9)


# auxiliary functions

def generate_fixed_lt_dataset(dataset_name, life_threshold) -> None:
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # folders
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    lt_dt_pairs = [(life_threshold, i) for i in range(0, 9+1)]
    n_subset = 30
    
    # individuals with fixed lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_LD_individual(
        id=id,
        life_threshold=lt_dt_pairs[pair][0], 
        death_threshold=lt_dt_pairs[pair][1], 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/ca_pair_{pair}_{id}',
        ) for pair in range(len(lt_dt_pairs)) for id in range(n_subset)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder)


def generate_fixed_dt_dataset(dataset_name, death_threshold) -> None:
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # folders
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    lt_dt_pairs = [(i, death_threshold) for i in range(0, 9+1)]
    n_subset = 30
    
    # individuals with fixed lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_LD_individual(
        id=id,
        life_threshold=lt_dt_pairs[pair][0], 
        death_threshold=lt_dt_pairs[pair][1], 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/ca_pair_{pair}_{id}',
        ) for pair in range(len(lt_dt_pairs)) for id in range(n_subset)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder)
