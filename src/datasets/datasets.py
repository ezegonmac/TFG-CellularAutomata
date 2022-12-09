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
    
    life_thresholds = []
    death_thresholds = []
    sizes = []
    initial_densities = []
    seeds = []
    iterations = []
    files = []
    for subset in subsets:
        # subset attributes
        subset_name = subset['name']
        subset_life_threshold = subset['lt']
        subset_death_threshold = subset['dt']
        subset_size = subset['size']
        subset_density = subset['density']
        subset_n_seeds = subset['n_seeds']
        subset_n_iterations = subset['iterations']

        # subset folder
        subset_folder = f'{DATA_FOLDER}/{dataset_name}/{subset_name}'
        if not os.path.exists(subset_folder):
            os.makedirs(subset_folder)
        
        # create subset attributes for every seed
        subset_seeds = range(0, subset_n_seeds)
        subset_life_thresholds = [subset_life_threshold] * subset_n_seeds
        subset_death_thresholds = [subset_death_threshold] * subset_n_seeds
        subset_sizes = [subset_size] * subset_n_seeds
        subset_densities = [subset_density] * subset_n_seeds
        subset_iterations = [subset_n_iterations] * subset_n_seeds
        subset_files = [f"{subset_folder}/ca_s{seed}" for seed in subset_seeds]
        
        save_subset_files_LD(
            subset_seeds, 
            subset_life_thresholds, 
            subset_death_thresholds, 
            subset_sizes, 
            subset_densities,
            subset_iterations,
            subset_files
            )

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

# not working
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
    
    subsets = [
        ['l1_d8', 1, 8],
        ['l3_d6', 3, 6],
        ['l5_d5', 5, 5],
        ['l6_d3', 6, 3],
        ['l8_d1', 8, 1],
    ]
    
    n_seeds = 4
    
    for subset in subsets:
        subset_folder = subset[0]
        life_threshold = subset[1]
        death_threshold = subset[2]
        
        save_file_LD(dataset_name, n_seeds, subset_folder, life_threshold, death_threshold)

