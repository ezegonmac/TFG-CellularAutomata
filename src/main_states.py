from constants import *
from statistics.states import *


if __name__ == '__main__':
    
    create_states_plot(DATASET8, individual=0, min_iteration=0, max_iteration=2)
    create_states_plot(DATASET8, individual=30, min_iteration=0, max_iteration=2)

    for i in range(0, 20):
        create_states_plot(DATASET3, individual=i, min_iteration=0, max_iteration=4)

    for i in range(0, 10):
        create_states_plot(DATASET9, individual=i, min_iteration=0, max_iteration=4)

    create_state_plot(DATASET8, individual=0, iteration=0)
    # create_state_plot(DATASET8, individual=0, iteration=1)
    # create_state_plot(DATASET8, individual=0, iteration=2)

    create_state_plot(DATASET8, individual=1, iteration=0)
    # create_state_plot(DATASET8, individual=20, iteration=1)
    # create_state_plot(DATASET8, individual=20, iteration=2)

    # create_state_plot(DATASET3, individual=0, iteration=0)
    # create_state_plot(DATASET3, individual=0, iteration=1)
    # create_state_plot(DATASET3, individual=0, iteration=2)
    # create_state_plot(DATASET3, individual=0, iteration=3)

    # create_state_plot(DATASET3, individual=1, iteration=0)
    # create_state_plot(DATASET3, individual=1, iteration=1)
    # create_state_plot(DATASET3, individual=1, iteration=2)
    # create_state_plot(DATASET3, individual=1, iteration=3)

    # create_state_plot(DATASET3, individual=2, iteration=0)
    # create_state_plot(DATASET3, individual=2, iteration=1)
    # create_state_plot(DATASET3, individual=2, iteration=2)
    # create_state_plot(DATASET3, individual=2, iteration=3)
