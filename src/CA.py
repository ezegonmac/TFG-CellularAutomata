import matplotlib.pyplot as plt
import random
import numpy as np
from RulesStrategies import RulesStrategy

class CA:
    
    def __init__(self, rules_strategy: RulesStrategy, size=None, density=None, board=None, iterations=10, **kwargs):
        
        # GENERAL PARAMETERS
        self._rules_strategy = rules_strategy
        self.iterations = iterations
        
        # BOARD
        if board is None:
            self.size = size
            self.density = density
            
            self.board = np.array([
                            [1 if random.uniform(0, 1) <= self.density else 0
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
        
    def __str__(self):
        return f"{self.board}"
    
    def print(self):
        print(self.board)
    
    def draw(self):
        plt.matshow(self.board, cmap='Greys', vmin=0, vmax=1)
        plt.show()
    
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
