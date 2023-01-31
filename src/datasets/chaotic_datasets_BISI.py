import numpy as np

from constants import *
from datasets.dataset_generation import generate_dataset_files_from_individuals
from datasets.CA_individual import CA_individual_state
from utils import *
from datasets.dataset_chaotic_generation import generate_initial_state, alter_initial_state

"""
`chaotic` datasets are made for studying the chaotic 
behavior of the cellular automata, generating minor changes in the initial state.
Composed of *K* clusters of *N* individuals.
Every cluster has the same rule, but different initial states.
The initial state is changed minimally with a *C* number of cells flipped.
"""

def generate_dataset11_chaotic(save_individuals=False) -> None:
    """
    Generate dataset11 chaotic.
    
    Description:
    Composed of *K* clusters of *N* individuals.
    Every cluster has the same rule, but different initial states.
    The initial state is changed minimally with a *C* number of cells flipped.
    
    Variables:
    - fixed size: 10x10
    - fixed density: 0.5
    - fixed iterations: 10
    - free B: (bmin, bmax) with bmin, bmax in [0, 9]
    - free S: (smin, smax) with smin, smax in [0, 9]
    """
    
    # subsets with all attributes
    dataset_name = DATASET11_CHAOTIC
    
    # fixed attributes
    size = 10
    density = 0.5
    iterations = 10
    
    # free attributes
    B_min = 0
    B_max = 8
    S_min = 0
    S_max = 8
    
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
    individuals = []
    for k in range(K):
        
        np.random.seed(NP_RANDOM_SEED + k)
        
        b1 = np.random.randint(B_min, B_max+1)
        b2 = np.random.randint(B_min, B_max+1)
        B = (b1, b2) if b1 < b2 else (b2, b1)
        
        s1 = np.random.randint(S_min, S_max+1)
        s2 = np.random.randint(S_min, S_max+1)
        S = (s1, s2) if s1 < s2 else (s2, s1)
        
        initial_state = generate_initial_state(size, density)
        
        for n in range(N):
            altered_initial_state = alter_initial_state(initial_state, C, size)
            
            ca = CA_individual_state(
                id=id,
                B=B,
                S=S,
                initial_state=altered_initial_state,
                iterations=iterations, 
                file=f'{individuals_folder}/ca_{id}_k{k}_n{n}',
                )
            
            individuals.append(ca)
            
    generate_dataset_files_from_individuals(
        individuals, dataset_folder, 
        individuals_folder, 
        save_individuals=save_individuals, 
        from_initial_state=True,
        rule_type='BISI')


def generate_dataset12_chaotic(save_individuals=False) -> None:
    """
    Generate dataset12 chaotic.
    
    Description:
    Composed of *K* clusters of *N* individuals.
    Every cluster has the same rule, but different initial states.
    The initial state is changed minimally with a *C* number of cells flipped.
    
    Variables:
    - fixed size: 10x10
    - fixed iterations: 10
    - free B: (bmin, bmax) with bmin, bmax in [0, 9]
    - free S: (smin, smax) with smin, smax in [0, 9]
    - free density: [0, 1]
    """
    
    # subsets with all attributes
    dataset_name = DATASET12_CHAOTIC
    
    # fixed attributes
    size = 10
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
    individuals = []
    for k in range(K):
        
        np.random.seed(NP_RANDOM_SEED + k)
        
        b1 = np.random.randint(B_min, B_max+1)
        b2 = np.random.randint(B_min, B_max+1)
        B = (b1, b2) if b1 < b2 else (b2, b1)
        
        s1 = np.random.randint(S_min, S_max+1)
        s2 = np.random.randint(S_min, S_max+1)
        S = (s1, s2) if s1 < s2 else (s2, s1)
        
        density = np.random.random()
        initial_state = generate_initial_state(size, density)
        
        for n in range(N):
            altered_initial_state = alter_initial_state(initial_state, C, size)
            
            ca = CA_individual_state(
                id=id,
                B=B,
                S=S,
                initial_state=altered_initial_state,
                iterations=iterations, 
                file=f'{individuals_folder}/ca_{id}_k{k}_n{n}',
                )
            
            individuals.append(ca)
            
    generate_dataset_files_from_individuals(
        individuals, 
        dataset_folder, 
        individuals_folder, 
        save_individuals=save_individuals, 
        from_initial_state=True,
        rule_type='BISI')
