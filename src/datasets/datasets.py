from datasets.LD_datasets import generate_dataset_LD
import numpy as np
from constants import *

def generate_dataset1() -> None:
    """
    Generate dataset1.
    
    Description:
    All cells dies or become alive in the next iteration.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 3
    - fixed life_threshold and death_threshold: (0, 9), (9, 0)
    """
    
    dataset_name = 'dataset1'
    
    # fixed attributes
    size = 10
    density = 0.5
    n_seeds = 10
    n_iterations = 3
    # variable attributes
    subsets = [
        {'name': 'all_live', 'lt': 0, 'dt': 9, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'all_die', 'lt': 9, 'dt': 0, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
    ]
    
    generate_dataset_LD(dataset_name, subsets)

def generate_dataset2():
    """
    Some cells die some become alive.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 3
    - fixed life_threshold and death_threshold: (1, 8), (3, 6), (5, 5), (6, 3), (8, 1)
    """
    dataset_name = 'dataset2'
    
    # fixed attributes
    size = 10
    density = 0.5
    n_seeds = 4
    n_iterations = 5
    # variable attributes
    subsets = [
        {'name': 'l1_d8', 'lt': 1, 'dt': 8, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'l3_d6', 'lt': 3, 'dt': 6, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'l5_d5', 'lt': 5, 'dt': 5, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'l6_d3', 'lt': 6, 'dt': 3, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'l8_d1', 'lt': 8, 'dt': 1, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
    ]
    
    generate_dataset_LD(dataset_name, subsets)

def generate_dataset3():
    """
    Some cells die some become alive.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 3
    - free life_threshold : [0, 9]
    - free death_threshold: [0, 9]
    """
    dataset_name = 'dataset3'
    
    # fixed attributes
    size = 10
    density = 0.5
    n_seeds = 1
    n_iterations = 10
    
    # free attributes
    lt_min = 0
    lt_max = 9
    dt_min = 0
    dt_max = 9
    
    # subsets with all attributes
    np.random.seed(NP_RANDOM_SEED)
    n_subsets = 50
    life_thesholds = np.random.randint(lt_min, lt_max, n_subsets)
    death_thesholds = np.random.randint(dt_min, dt_max, n_subsets)
    
    subsets = [
        {'name': f'l{lt}_d{dt}', 'lt': lt, 'dt': dt, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations}
        for lt, dt in zip(life_thesholds, death_thesholds) 
    ]
    
    generate_dataset_LD(dataset_name, subsets)
