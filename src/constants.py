NP_RANDOM_SEED = 0
SKLEARN_RANDOM_SEED = 0

#----------------#
#    FOLDERS     #
#----------------#
# DATA
DATA_FOLDER = './data'
#  - Learning
DATA_LEARNING_FOLDER = f'{DATA_FOLDER}/learning'
#  - Datasets
DATA_DATASETS_FOLDER = f'{DATA_FOLDER}/datasets'

# FIGURES
FIGURES_FOLDER = './figures'


#----------------#
#    DATASETS    #
#----------------#
# datasets
#  - BTST
DATASET1 = 'dataset1'
DATASET2 = 'dataset2'
DATASET3 = 'dataset3'
DATASET4 = 'dataset4'
DATASET5 = 'dataset5'
DATASET6 = 'dataset6'
DATASET7 = 'dataset7'
DATASET8 = 'dataset8'  # dataset 1 en el documento
DATASET9 = 'dataset9'
DATASET10 = 'dataset10'
#  - BISI
DATASET11 = 'dataset11'
DATASET12 = 'dataset12'
DATASET13 = 'dataset13'

# density datasets
# - BTST
DATASET8_DENSITY = DATASET8 + '_density'
DATASET3_DENSITY = DATASET3 + '_density'
DATASET9_DENSITY = DATASET9 + '_density'
DATASET10_DENSITY = DATASET10 + '_density'
# - BISI
DATASET11_DENSITY = DATASET11 + '_density'
DATASET12_DENSITY = DATASET12 + '_density'
DATASET13_DENSITY = DATASET13 + '_density'

# datasets clasification
DATASETS_BY_TYPE = {
    DATASET1: 'BTST',
    DATASET2: 'BTST',
    DATASET3: 'BTST',
    DATASET4: 'BTST',
    DATASET5: 'BTST',
    DATASET6: 'BTST',
    DATASET7: 'BTST',
    DATASET8: 'BTST',
    DATASET9: 'BTST',
    DATASET10: 'BTST',
    DATASET8_DENSITY: 'BTST',
    DATASET3_DENSITY: 'BTST',
    DATASET9_DENSITY: 'BTST',
    DATASET10_DENSITY: 'BTST',
    
    DATASET11: 'BISI',
    DATASET11_DENSITY: 'BISI',
    DATASET12: 'BISI',
    DATASET12_DENSITY: 'BISI',
    DATASET13: 'BISI',
    DATASET13_DENSITY: 'BISI',
}


#----------------#
#     STYLES     #
#----------------#
# colors and colormaps
COLOR_BLUE = 'blue'  # '#0061FF'
COLOR_PRIMARY = COLOR_BLUE
