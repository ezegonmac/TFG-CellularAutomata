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
DATASET8 = 'dataset8'
DATASET9 = 'dataset9'
DATASET10 = 'dataset10'
#  - BISI
DATASET11 = 'dataset11'
DATASET11 = 'dataset11'
DATASET12 = 'dataset12'
DATASET13 = 'dataset13'
#  - BS
DATASET14 = 'dataset14'
DATASET15 = 'dataset15'

# density datasets
# - BTST
DATASET8_DENSITY = DATASET8 + '_density'
DATASET3_DENSITY = DATASET3 + '_density'
DATASET9_DENSITY = DATASET9 + '_density'
DATASET10_DENSITY = DATASET10 + '_density'
# - BISI
DATASET11_DENSITY = DATASET11 + '_density'
DATASET11_DENSITY = DATASET11 + '_density'
DATASET12_DENSITY = DATASET12 + '_density'
DATASET13_DENSITY = DATASET13 + '_density'

# chaotic datasets
# - BTST
DATASET3_CHAOTIC = DATASET3 + '_chaotic'
DATASET9_CHAOTIC = DATASET9 + '_chaotic'
DATASET11_CHAOTIC = DATASET11 + '_chaotic'
DATASET12_CHAOTIC = DATASET12 + '_chaotic'

# datasets clasification
DATASETS_BY_TYPE = {
    DATASET1: 'BTST',
    DATASET2: 'BTST',
    DATASET3: 'BTST',
    DATASET3_DENSITY: 'BTST',
    DATASET3_CHAOTIC: 'BTST',
    DATASET4: 'BTST',
    DATASET5: 'BTST',
    DATASET6: 'BTST',
    DATASET7: 'BTST',
    DATASET8: 'BTST',
    DATASET8_DENSITY: 'BTST',
    DATASET9: 'BTST',
    DATASET9_DENSITY: 'BTST',
    DATASET9_CHAOTIC: 'BTST',
    DATASET10: 'BTST',
    DATASET10_DENSITY: 'BTST',
    
    DATASET11: 'BISI',
    DATASET11_DENSITY: 'BISI',
    DATASET11: 'BISI',
    DATASET11_DENSITY: 'BISI',
    DATASET11_CHAOTIC: 'BISI',
    DATASET12: 'BISI',
    DATASET12_DENSITY: 'BISI',
    DATASET12_CHAOTIC: 'BISI',
    DATASET13: 'BISI',
    DATASET13_DENSITY: 'BISI',
    
    DATASET14: 'BS',
    DATASET15: 'BS',
}

#----------------#
#     MODELS     #
#----------------#
KNN = 'KNN'
DECISION_TREE = 'DecisionTree'
RANDOM_FOREST = 'RandomForest'
NEURAL_NETWORK = 'NeuralNetwork'


#----------------#
#     STYLES     #
#----------------#
# colors and colormaps
COLOR_BLUE = 'blue'  # '#0061FF'
COLOR_PRIMARY = COLOR_BLUE
