from CA.CAFactory import CAFactory
import os
import numpy as np
import pandas as pd

DATA_FOLDER = './data'

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
    n_iterations = 3
    # variable attributes
    subsets = [
        {'name': 'l1_d8', 'lt': 1, 'dt': 8, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'l3_d6', 'lt': 3, 'dt': 6, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'l5_d5', 'lt': 5, 'dt': 5, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'l6_d3', 'lt': 6, 'dt': 3, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
        {'name': 'l8_d1', 'lt': 8, 'dt': 1, 'size': size, 'density': density, 'n_seeds': n_seeds, 'iterations': n_iterations},
    ]
    
    generate_dataset_LD(dataset_name, subsets)

def generate_dataset_LD(dataset_name, subsets):
    
    life_thresholds = []
    death_thresholds = []
    sizes = []
    initial_densities = []
    seeds = []
    iterations = []
    files = []
    for subset in subsets:
        [subset_seeds,
         subset_life_thresholds,
         subset_death_thresholds,
         subset_sizes, subset_densities,
         subset_iterations,
         subset_files] = generate_subset_LD(dataset_name, subset)
        
        # add subset attributes to lists
        seeds.extend(subset_seeds)
        life_thresholds.extend(subset_life_thresholds)
        death_thresholds.extend(subset_death_thresholds)
        sizes.extend(subset_sizes)
        initial_densities.extend(subset_densities)
        iterations.extend(subset_iterations)
        files.extend(subset_files)

    df = pd.DataFrame(
        data={'seed' : seeds,
              'life_threshold' : life_thresholds, 
              'death_threshold' : death_thresholds, 
              'size' : sizes,
              'initial_density' : initial_densities,
              'iterations' : iterations,
              'file' : files})
    
    df.to_csv(f'{DATA_FOLDER}/{dataset_name}/dataset.csv')

def generate_subset_LD(dataset_name, subset):
    
    # subset attributes
    name = subset['name']
    life_threshold = subset['lt']
    death_threshold = subset['dt']
    size = subset['size']
    density = subset['density']
    n_seeds = subset['n_seeds']
    n_iterations = subset['iterations']

    # subset folder
    folder = f'{DATA_FOLDER}/{dataset_name}/{name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # create subset attributes for every seed
    seeds = range(0, n_seeds)
    life_thresholds = [life_threshold] * n_seeds
    death_thresholds = [death_threshold] * n_seeds
    sizes = [size] * n_seeds
    densities = [density] * n_seeds
    iterations = [n_iterations] * n_seeds
    files = [f"{folder}/ca_s{seed}" for seed in seeds]
    
    save_subset_files_LD(
        seeds, 
        life_thresholds, 
        death_thresholds, 
        sizes, 
        densities,
        iterations,
        files
        )
    
    return [seeds, life_thresholds, death_thresholds, sizes, densities, iterations, files]

def save_subset_files_LD(seeds, life_thresholds, death_thresholds, sizes, densities, iterations, files) -> None:    
    
    attributes = zip(seeds, life_thresholds, death_thresholds, sizes, densities, iterations, files)
    for seed, life_threshold, death_threshold, size, density, n_iterations, file in attributes:
        # different seed for every iteration
        np.random.seed(seed)
        
        ca1 = CAFactory.create_CA_LB(
            life_threshold=life_threshold,
            death_threshold=death_threshold,
            size=size,
            density=density,
            iterations=n_iterations)
        
        ca1.save_evolution(file)
