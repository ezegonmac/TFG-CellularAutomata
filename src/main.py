from constants import *
from datasets.datasets import *
from statistics.densities import *

if __name__ == '__main__':

    # create_dataset_density_evolution_plot(DATASET1, show=True)
    # create_dataset_density_evolution_plot(DATASET1)
    # create_dataset_density_evolution_plot(DATASET2)
    # create_dataset_density_evolution_plot(DATASET3)
    
    generate_dataset4()
    create_dataset_density_evolution_plot(DATASET4)
    generate_dataset5()
    create_dataset_density_evolution_plot(DATASET5)
    