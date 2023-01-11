import matplotlib.pyplot as plt
import numpy as np
from CA.RulesStrategies import RulesStrategy
import time

SLEEP_PLOT_TIME = 1

class CA:
    
    def __init__(self, rules_strategy: RulesStrategy, iterations:int = 10, size:int = None, density:int = None, board:np.array = None, **kwargs):
        """Cellular Automata object. It can be used to compute the evolution of a CA board with a specified rule strategy.

        Args:
            rules_strategy (RulesStrategy): rules strategy to be used for cell update function
            iterations (int, optional): number of iterations to be conmputed. Defaults to 10.
            
            If you want the board to be generated automatically: 
            size (int, optional): width and height of the board. Defaults to None.
            density (int, optional): initial configuration average density. Defaults to None.
            
            If you want to specify the board:
            board (np.array, optional): specify initial configuration board. Defaults to None.
        Extra args (depending on rules_strategy specified):
            B (list, tuple or int): (Birth)
            S (list, tuple or int): (Survival)
        Raises:
            ValueError: Board and size cannot be both specified. When size is specified the board is auto generated.
            ValueError: Board and density cannot be both specified. When density is specified the board is auto generated.
        """
        
        # GENERAL PARAMETERS
        self._rules_strategy = rules_strategy
        self.iterations = iterations
        
        # BOARD
        if board is None:
            self.size = size
            self.density = density
            
            self.board = np.array([
                            [1 if np.random.uniform(0, 1) <= self.density else 0
                            for i in range(0, self.size)]
                            for j in range(0, self.size)
                        ], dtype=np.int8)
        else:
            if size is not None:
                raise ValueError('Board and size cannot be both specified. When size is specified the board is auto generated.')
            if density is not None:
                raise ValueError('Board and density cannot be both specified. When density is specified the board is auto generated.')
            self.board = board
            self.size = board.shape[0]
        
        # EXTRA PARAMETERS (RULES, ...)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # STATE evolution
        self.evolution = np.array([self.board])
        if(self.iterations > 0):
            for i in range(0, self.iterations-1):
                self.update()
                self.evolution = np.append(self.evolution, [self.board], axis=0)
    
    def __str__(self):
        return f"{self.board}"
    
    def print(self):
        print(self.board)
        
    def print_evolution(self):
        print(self.evolution)
    
    def get_neighbours(self, i, j):
        count = 0
        for r in range(i-1, i+2):
            for c in range(j-1, j+2):
                r = 0 if r == self.size else r
                c = 0 if c == self.size else c
                
                if not (r == i and c == j):
                    count += self.board[r][c]
        
        return count

    def update(self):
        new_board = np.zeros((self.size, self.size), dtype=np.int8)
        for i in range(0, self.size):
            for j in range(0, self.size):
                cell = self.board[i][j]
                self._rules_strategy.update_cell(self, new_board, cell, i, j)

        self.board = new_board
    
    def get_density_evolution(self):
        density_evolution = np.zeros(self.iterations)
        for i in range(self.iterations):
            density_evolution[i] = np.count_nonzero(self.evolution[i]) / (self.size * self.size)

        return density_evolution
        
    def draw(self):
        plt.matshow(self.board, cmap='Greys', vmin=0, vmax=1)
        plt.show()
    
    def draw_evolution(self):
        plt.ion()
        
        # fig = plt.figure()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        graph = ax.matshow(self.evolution[0], cmap='Greys', vmin=0, vmax=1)
        
        for i in range(0, self.evolution.shape[0]):
            # updating data values
            graph.set_data(self.evolution[i])

            fig.canvas.draw()
            
            fig.canvas.flush_events()
        
            time.sleep(SLEEP_PLOT_TIME)

    def save_evolution(self, filename):
        np.save(filename, self.evolution)
