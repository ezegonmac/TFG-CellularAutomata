import numpy as np


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
    alter_initial_state = initial_state.copy()
    
    flipped_cells_x = np.random.randint(0, size, C)
    flipped_cells_y = np.random.randint(0, size, C)

    for x, y in zip(flipped_cells_x, flipped_cells_y):
        state = alter_initial_state[x, y]
        alter_initial_state[x, y] = 1 if state == 0 else 0
    
    return alter_initial_state
