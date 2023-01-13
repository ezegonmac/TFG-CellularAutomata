from constants import *
from statistics.densities import *


if __name__ == '__main__':

    # create_dataset_density_evolution_per_S_plot(DATASET1)
    # create_dataset_density_evolution_per_B_plot(DATASET1)
    # print('dataset1 done')
    
    # create_dataset_density_evolution_plot(DATASET3)
    # create_dataset_density_evolution_per_S_plot(DATASET3)
    # create_dataset_density_evolution_per_B_plot(DATASET3)
    # create_dataset_density_evolution_plot(DATASET3)
    # create_dataset_density_variation_plot(DATASET3)
    # create_dataset_density_histograms_plot(DATASET3)
    # print('dataset3 done')
    
    # create_dataset_density_evolution_per_S_plot(DATASET4)
    # create_dataset_density_evolution_per_B_plot(DATASET4)
    # print('dataset4 done')
    
    # create_dataset_density_evolution_per_S_plot(DATASET5)
    # create_dataset_density_evolution_per_B_plot(DATASET5)
    # print('dataset5 done')
    
    # create_dataset_density_evolution_per_S_plot(DATASET6)
    # create_dataset_density_evolution_per_B_plot(DATASET6)
    # print('dataset6 done')
    
    # create_dataset_density_evolution_per_S_plot(DATASET7)
    # create_dataset_density_evolution_per_B_plot(DATASET7)
    # print('dataset7 done')
    
    # create_dataset_density_evolution_per_S_plot(DATASET8)
    # create_dataset_density_evolution_per_B_plot(DATASET8)
    # create_dataset_density_evolution_per_B_and_S_plot(DATASET8)
    # print('dataset8 done')
    
    create_dataset_density_evolution_plot(DATASET9, limit=500)
    create_dataset_density_evolution_per_S_plot(DATASET9, limit=500)
    create_dataset_density_evolution_per_B_plot(DATASET9, limit=500)
    create_dataset_density_evolution_plot(DATASET9, limit=500)
    create_dataset_density_variation_plot(DATASET9, limit=500)
    create_dataset_density_histograms_plot(DATASET9, limit=500)
    print('dataset9 done')
