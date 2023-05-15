from constants import *
from statistics.densities import *
from statistics.histograms import *
from statistics.chaotic import *


if __name__ == '__main__':

    # create_dataset_density_evolution_per_S_plot(DATASET1)
    # create_dataset_density_evolution_per_B_plot(DATASET1)
    # print('dataset1 done')
    
    dataset = DATASET3
    # create_dataset_density_evolution_plot(dataset, limit=300)
    # create_dataset_density_variation_plot(dataset)
    # create_dataset_density_evolution_per_S_plot(dataset, limit=300)
    # create_dataset_density_evolution_per_B_plot(dataset, limit=300)
    # create_dataset_density_histograms_plot(dataset, limit=500)
    # print(dataset + 'done')
    
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
    
    dataset = DATASET8
    # create_dataset_density_evolution_per_S_plot(DATASET8)
    # create_dataset_density_evolution_per_B_plot(DATASET8)
    # create_dataset_density_evolution_per_B_and_S_plot(DATASET8, limit=300)
    # print('dataset8 done')
    
    dataset = DATASET9
    # create_dataset_density_evolution_plot(dataset, limit=500)
    # create_dataset_density_variation_plot(dataset, limit=500)
    # create_dataset_density_evolution_per_S_plot(dataset, limit=500)
    # create_dataset_density_evolution_per_B_plot(dataset, limit=500)
    # create_dataset_density_histograms_plot(dataset, limit=500)
    # print('dataset9 done')
    
    dataset = DATASET10
    # create_dataset_density_evolution_plot(dataset, limit=500)
    # create_dataset_density_evolution_per_S_plot(dataset, limit=500)
    # create_dataset_density_evolution_per_B_plot(dataset, limit=500)
    # create_dataset_density_variation_plot(dataset, limit=500)
    # print('dataset10 done')
    
    dataset = DATASET11
    # create_dataset_density_evolution_plot(dataset, limit=10, suffix='10', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=100, limit=10, suffix='10_1', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=200, limit=10, suffix='10_2', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=300, limit=10, suffix='10_3', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=400, limit=10, suffix='10_4', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=500, limit=10, suffix='10_5', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, limit=100, suffix='100')
    # create_dataset_density_evolution_plot(dataset, limit=200, suffix='200')
    # create_dataset_density_evolution_plot(dataset, limit=500, suffix='500')
    # create_dataset_density_evolution_plot(dataset, limit=700, suffix='700')
    # create_dataset_density_variation_plot(dataset, limit=200, suffix='200_2')
    # create_dataset_density_variation_plot(dataset, limit=500, suffix='500_2')
    # create_dataset_density_histograms_plot(dataset)
    # print('dataset11 done')
    
    dataset = DATASET12
    # create_dataset_density_evolution_plot(dataset, limit=10, suffix='10', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=100, limit=10, suffix='10_1', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=200, limit=10, suffix='10_2', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=300, limit=10, suffix='10_3', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=400, limit=10, suffix='10_4', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=500, limit=10, suffix='10_5', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, limit=100, suffix='100')
    # create_dataset_density_evolution_plot(dataset, limit=200, suffix='200')
    # create_dataset_density_evolution_plot(dataset, limit=500, suffix='500')
    # create_dataset_density_evolution_plot(dataset, limit=700, suffix='700')
    # create_dataset_density_variation_plot(dataset, limit=200, suffix='200_2')
    # create_dataset_density_variation_plot(dataset, limit=500, suffix='500_2')
    create_dataset_density_histograms_plot(dataset)
    # print(f'{dataset} done')
    
    dataset = DATASET13
    # create_dataset_density_evolution_plot(dataset, limit=10, suffix='10_0', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=100, limit=10, suffix='10_1', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=200, limit=10, suffix='10_2', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=300, limit=10, suffix='10_3', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=400, limit=10, suffix='10_4', num_colors=10)
    # create_dataset_density_evolution_plot(dataset, start=500, limit=10, suffix='10_5', num_colors=10)
    create_dataset_density_evolution_plot(dataset, start=400, limit=50, suffix='50')
    # create_dataset_density_evolution_plot(dataset, limit=100, suffix='100')
    # create_dataset_density_evolution_plot(dataset, limit=200, suffix='200')
    # create_dataset_density_evolution_plot(dataset, limit=500, suffix='500')
    # create_dataset_density_evolution_plot(dataset, limit=700, suffix='700')
    # create_dataset_density_variation_plot(dataset, limit=200, suffix='200_2')
    # create_dataset_density_variation_plot(dataset, limit=500, suffix='500_2')
    # print(f'{dataset} done')
    
    # dataset = DATASET14
    # create_dataset_density_evolution_plot(dataset, limit=10, suffix='10_1', start=100, num_colors=10)
    # create_dataset_density_evolution_plot(dataset, limit=10, suffix='10_2', start=1000, num_colors=10)
    # create_dataset_density_evolution_plot(dataset, limit=50, suffix='50')
    # create_dataset_density_evolution_plot(dataset, limit=100, suffix='100')
    # create_dataset_density_evolution_plot(dataset, limit=200, suffix='200')
    # create_dataset_density_evolution_plot(dataset, limit=500, suffix='500')
    # create_dataset_density_variation_plot(dataset, limit=200, suffix='200')
    # create_dataset_density_variation_plot(dataset, limit=500, suffix='500')
    # print(f'{dataset} done')
    
    # dataset = DATASET15
    # create_dataset_density_evolution_plot(dataset, limit=10, suffix='10_1', start=100, num_colors=10)
    # create_dataset_density_evolution_plot(dataset, limit=10, suffix='10_2', start=300, num_colors=10)
    # create_dataset_density_evolution_plot(dataset, limit=10, suffix='10_3', start=500, num_colors=10)
    # create_dataset_density_evolution_plot(dataset, limit=50, suffix='50')
    # create_dataset_density_evolution_plot(dataset, limit=100, suffix='100')
    # create_dataset_density_evolution_plot(dataset, limit=200, suffix='200')
    # create_dataset_density_evolution_plot(dataset, limit=500, suffix='500')
    # create_dataset_density_variation_plot(dataset, limit=200, suffix='200')
    # create_dataset_density_variation_plot(dataset, limit=500, suffix='500')
    # print(f'{dataset} done')
