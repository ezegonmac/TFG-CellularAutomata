from constants import *
from statistics.states import *


if __name__ == '__main__':

    create_state_plot(DATASET1, individual=0, iteration=0)
    create_state_plot(DATASET1, individual=0, iteration=1)
    create_state_plot(DATASET1, individual=0, iteration=2)

    create_state_plot(DATASET1, individual=20, iteration=0)
    create_state_plot(DATASET1, individual=20, iteration=1)
    create_state_plot(DATASET1, individual=20, iteration=2)

    create_state_plot(DATASET3, individual=0, iteration=0)
    create_state_plot(DATASET3, individual=0, iteration=1)
    create_state_plot(DATASET3, individual=0, iteration=2)
    create_state_plot(DATASET3, individual=0, iteration=3)

    create_state_plot(DATASET3, individual=50, iteration=0)
    create_state_plot(DATASET3, individual=50, iteration=1)
    create_state_plot(DATASET3, individual=50, iteration=2)
    create_state_plot(DATASET3, individual=50, iteration=3)

    create_state_plot(DATASET3, individual=120, iteration=0)
    create_state_plot(DATASET3, individual=120, iteration=1)
    create_state_plot(DATASET3, individual=120, iteration=2)
    create_state_plot(DATASET3, individual=120, iteration=3)
