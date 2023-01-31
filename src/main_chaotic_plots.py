from constants import *
from statistics.chaotic import *


if __name__ == '__main__':

    # create_dataset_chaotic_evolution_plots(DATASET3_CHAOTIC, 
    #                                        clusters=10, 
    #                                        n_per_cluster=100)
    # print('dataset3 done')

    # create_dataset_chaotic_evolution_plots(DATASET9_CHAOTIC, 
    #                                        clusters=10, 
    #                                        n_per_cluster=100)
    # print('dataset9 done')

    create_dataset_chaotic_evolution_plots(DATASET11_CHAOTIC, 
                                           clusters=10, 
                                           n_per_cluster=100)
    print('dataset11 done')

    create_dataset_chaotic_evolution_plots(DATASET12_CHAOTIC, 
                                           clusters=10, 
                                           n_per_cluster=100)
    print('dataset12 done')
