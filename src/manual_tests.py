import numpy as np
from CA import CA

def test_1():
    board = np.array([
        [0,1,0],
        [1,0,1],
        [0,1,0]
    ])
    ca1 = CA(board=board)
    ca1.print()
    ca1.draw()
    
    ca1.update()
    ca1.print()
    ca1.draw()

def test_2():
    ca1 = CA(size=10, density=0.7, life_threshold=3, death_threshold=2)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()
        
def test_2_all_die():
    ca1 = CA(size=10, density=0.7, life_threshold=9, death_threshold=0)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()

def test_2_all_live():
    ca1 = CA(size=10, density=0.7, life_threshold=0, death_threshold=9)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()