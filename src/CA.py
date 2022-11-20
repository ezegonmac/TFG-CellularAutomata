import matplotlib.pyplot as plt
import random
import numpy as np

class CA:
    
    def __init__(self, size=30, density=0.5):
        self.size = size
        self.density = density
        self.board = np.array([
                        [1 if random.uniform(0, 1) <= self.density else 0
                        for i in range(0,self.size)]
                        for j in range(0,self.size)
                    ])
        
    def __str__(self):
        return f"{self.board}"
    
    def print(self):
        print(self.board)
    
    def draw(self):
        plt.matshow(self.board, cmap='Greys')
        plt.show()