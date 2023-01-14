import numpy as np

from constants import *
from datasets.dataset_generation import generate_dataset_files_from_individuals
from datasets.CA_individual import CA_individual
from utils import *


def generate_dataset1(save_individuals=False) -> None:
    """
    Generate dataset1.
    
    Description:
    All cells dies or become alive in the next iteration.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (0, 9), (9, 0)
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
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # subsets
    np.random.seed(NP_RANDOM_SEED)
    subset1 = [CA_individual(
        id=id,
        B=0, 
        S=9, 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/all_live_{id}',
        ) for id in range(n_subset1)]
    
    subset2 = [CA_individual(
        id=id,
        B=9, 
        S=0, 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/all_die_{id}',
        ) for id in range(n_subset2)]

    subsets = [subset1, subset2]
    
    # individuals contains all subset individuals
    individuals = [individual for subset in subsets for individual in subset]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals)


def generate_dataset2(save_individuals=False) -> None:
    """
    Generate dataset2.
    
    Some cells die some become alive.
    Fixed rules pairs.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (1, 8), (3, 6), (5, 5), (6, 3), (8, 1)
    """
    
    dataset_name = DATASET2
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    lt_dt_pairs = [(1, 8), (3, 6), (5, 5), (6, 3), (8, 1)]
    n_subset = 30
    
    # individuals with fixed lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=lt_dt_pairs[pair][0], 
        S=lt_dt_pairs[pair][1], 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/ca_pair_{pair}_{id}',
        ) for pair in range(len(lt_dt_pairs)) for id in range(n_subset)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals)


def generate_dataset3(save_individuals=False) -> None:
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
      
    n_individuals = 100
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=np.random.randint(lt_min, lt_max+1), 
        S=np.random.randint(dt_min, dt_max+1), 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/ca_{id}',
        ) for id in range(n_individuals)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals)


def generate_dataset4(save_individuals=False) -> None:
    """
    Generate dataset4.
    
    Some cells die some become alive.
    Fixed life threshold.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (0, 0), (0, 1), ... (0, 9)
    """
    
    dataset_name = DATASET4
    
    generate_fixed_lt_dataset(dataset_name, 0)
    

def generate_dataset5(save_individuals=False) -> None:
    """
    Generate dataset5.
    
    Some cells die some become alive.
    Fixed death threshold.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (0, 0), (1, 0), ... (9, 0)
    """
    
    dataset_name = DATASET5
    
    generate_fixed_dt_dataset(dataset_name, 0)


def generate_dataset6(save_individuals=False) -> None:
    """
    Generate dataset6.
    
    Some cells die some become alive.
    Fixed life threshold.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (9, 0), (9, 1), ... (9, 9)
    """
    
    dataset_name = DATASET6
    
    generate_fixed_lt_dataset(dataset_name, 9)
    

def generate_dataset7(save_individuals=False) -> None:
    """
    Generate dataset7.
    
    Some cells die some become alive.
    Fixed death threshold.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (0, 9), (1, 9), ... (9, 9)
    """
    
    dataset_name = DATASET7
    
    generate_fixed_dt_dataset(dataset_name, 9)


# auxiliary functions

def generate_fixed_lt_dataset(dataset_name, B) -> None:
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    lt_dt_pairs = [(B, i) for i in range(0, 9+1)]
    n_subset = 30
    
    # individuals with fixed lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=lt_dt_pairs[pair][0], 
        S=lt_dt_pairs[pair][1], 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/ca_pair_{pair}_{id}',
        ) for pair in range(len(lt_dt_pairs)) for id in range(n_subset)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals)


def generate_fixed_dt_dataset(dataset_name, S) -> None:
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    lt_dt_pairs = [(i, S) for i in range(0, 9+1)]
    n_subset = 30
    
    # individuals with fixed lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=lt_dt_pairs[pair][0], 
        S=lt_dt_pairs[pair][1], 
        size=size, 
        density=density, 
        iterations=iterations, 
        file=f'{individuals_folder}/ca_pair_{pair}_{id}',
        ) for pair in range(len(lt_dt_pairs)) for id in range(n_subset)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals)


def generate_dataset8(save_individuals=False) -> None:
    """
    Generate dataset8.
    
    Description:
    All cells dies or become alive in the next iteration, for new rules.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - fixed B and S: (0, 0), (9, 9)
    """
    
    dataset_name = DATASET8
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # define subsets with same attributes
    n_subset1 = 20
    n_subset2 = 20
    
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


def generate_dataset9(save_individuals=False) -> None:
    """
    Generate dataset9.
    
    Description:
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
    dataset_name = DATASET9
    
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
    
    n_individuals = 100
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=np.random.randint(B_min, B_max+1), 
        S=np.random.randint(B_min, S_max+1), 
        size=size, 
        density=np.random.uniform(density_min, density_max),
        iterations=iterations, 
        file=f'{individuals_folder}/ca_{id}',
        ) for id in range(n_individuals)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals)


def generate_dataset10(save_individuals=False) -> None:
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
    dataset_name = DATASET10
    
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
    individuals_folder = f'{dataset_folder}/individuals'
    
    # individuals with random lt and dt
    np.random.seed(NP_RANDOM_SEED)
    individuals = [CA_individual(
        id=id,
        B=np.random.randint(B_min, B_max+1), 
        S=np.random.randint(B_min, S_max+1), 
        size=size, 
        density=np.random.uniform(density_min, density_max),
        iterations=iterations, 
        file=f'{individuals_folder}/ca_{id}',
        ) for id in range(n_individuals)]
    
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals)
