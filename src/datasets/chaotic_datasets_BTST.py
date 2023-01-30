import numpy as np

from constants import *
from datasets.dataset_generation import generate_dataset_files_from_individuals
from datasets.CA_individual import CA_individual_state
from utils import *

"""
`chaotic` datasets are made for studying the chaotic 
behavior of the cellular automata, generating minor changes in the initial state.
Composed of *K* clusters of *N* individuals.
Every cluster has the same rule, but different initial states.
The initial state is changed minimally with a *C* number of cells flipped.
"""

def generate_dataset3_chaotic(save_individuals=False) -> None:
    """
    Generate dataset3 chaotic.
    
    Description:
    Composed of *K* clusters of *N* individuals.
    Every cluster has the same rule, but different initial states.
    The initial state is changed minimally with a *C* number of cells flipped.
    
    Variables:
    - fixed size: 10x10
    - fixed iterations: 10
    - free B : [0, 9]
    - free S: [0, 9]
    - fixed density: 0.5
    """
    
    # subsets with all attributes
    dataset_name = DATASET3_CHAOTIC
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # free attributes
    B_min = 0
    B_max = 9
    S_min = 0
    S_max = 9
    
    # number of clusters
    K = 10
    # number of individuals per cluster
    N = 100
    # number of cells to flip
    C = 5
    
    # folders
    data_datasets_folder = get_data_datasets_folder(dataset_name)
    dataset_folder = f'{data_datasets_folder}/{dataset_name}'
    individuals_folder = f'{dataset_folder}/individuals'
    
    # individuals grouped by clusters
    np.random.seed(NP_RANDOM_SEED)
    individuals = []
    for k in range(K):
        
        B = np.random.randint(B_min, B_max+1)
        S = np.random.randint(S_min, S_max+1)
        
        initial_state = generate_initial_state(size, density)
        
        for n in range(N):
            initial_state = alter_initial_state(initial_state, C, size)
            
            ca = CA_individual_state(
                id=id,
                B=B,
                S=S,
                initial_state=initial_state,
                iterations=iterations, 
                file=f'{individuals_folder}/ca_{id}_k{k}_n{n}',
                )
            
            individuals.append(ca)
            
    generate_dataset_files_from_individuals(individuals, dataset_folder, individuals_folder, save_individuals=save_individuals, 
                                            from_initial_state=True)


def generate_initial_state(size, density):
    initial_state = np.array([
                            [1 if np.random.uniform(0, 1) <= density else 0
                            for i in range(0, size)]
                            for j in range(0, size)
                        ], dtype=np.int8)
                    
    return initial_state


def alter_initial_state(initial_state, C, size):
    """
    Alter the initial state by flipping C cells.
    """
    
    flipped_cells_x = np.random.randint(0, size, C)
    flipped_cells_y = np.random.randint(0, size, C)
    
    initial_state = initial_state.copy()
    for x, y in zip(flipped_cells_x, flipped_cells_y):
        state = initial_state[x, y]
        initial_state[x, y] = 1 if state == 0 else 0
    
    return initial_state
