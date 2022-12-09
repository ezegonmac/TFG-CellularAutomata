from CA.CAFactory import CAFactory
import os
import numpy as np
import pandas as pd
from shutil import rmtree
from constants import *

DATA_FOLDER = './data'

def generate_dataset_LD(dataset_name, subsets):
    np.random.seed(NP_RANDOM_SEED)
    
    # delete old dataset
    dataset_folder = f'{DATA_FOLDER}/{dataset_name}'
    if os.path.exists(dataset_folder):
        rmtree(dataset_folder)
    
    life_thresholds = []
    death_thresholds = []
    sizes = []
    densities = []
    seeds = []
    iterations = []
    files = []
    evolution_densities = []
    for subset in subsets:
        [subset_seeds,
         subset_life_thresholds,
         subset_death_thresholds,
         subset_sizes, subset_densities,
         subset_iterations,
         subset_files,
         subset_evolution_densities] = generate_subset_LD(dataset_name, subset)
        
        # add subset attributes to lists
        seeds.extend(subset_seeds)
        life_thresholds.extend(subset_life_thresholds)
        death_thresholds.extend(subset_death_thresholds)
        sizes.extend(subset_sizes)
        densities.extend(subset_densities)
        iterations.extend(subset_iterations)
        files.extend(subset_files)
        evolution_densities.extend(subset_evolution_densities)

    df = pd.DataFrame(
        data={'seed' : seeds,
              'life_threshold' : life_thresholds, 
              'death_threshold' : death_thresholds, 
              'size' : sizes,
              'density' : densities,
              'iterations' : iterations,
              'file' : files,
              'evolution_density' : evolution_densities,
              })
    
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
    folder = f'{DATA_FOLDER}/{dataset_name}/subsets/{name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # create subset attributes for every seed (precalculated)
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
    
    # create subset attributes for every seed (calculated from file)
    evolution_densities = [calc_evolution_densities_from_file(file) for file in files]
    
    return [seeds, life_thresholds, death_thresholds, sizes, densities, iterations, files, evolution_densities]

def save_subset_files_LD(seeds, life_thresholds, death_thresholds, sizes, densities, iterations, files) -> None:    
    
    attributes = zip(seeds, life_thresholds, death_thresholds, sizes, densities, iterations, files)
    for seed, life_threshold, death_threshold, size, density, n_iterations, file in attributes:
        # different seed for every iteration
        np.random.seed(NP_RANDOM_SEED + seed)
        
        ca1 = CAFactory.create_CA_LB(
            life_threshold=life_threshold,
            death_threshold=death_threshold,
            size=size,
            density=density,
            iterations=n_iterations)
        
        ca1.save_evolution(file)
        
def calc_evolution_densities_from_file(file):
    evolution_densities = []
    
    evolution = np.load(file + '.npy')
    
    iterations = evolution.shape[0]
    
    for it in range(0, iterations):
        state = evolution[it]
        evolution_densities.append(np.count_nonzero(state) / state.size)
    
    return evolution_densities
