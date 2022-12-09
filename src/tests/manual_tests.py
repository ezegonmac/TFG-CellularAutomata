import numpy as np
from CA.CA import CA
from CA.RulesStrategies import *
from CA.CAFactory import CAFactory

def test_1():
    board = np.array([
        [0,1,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,0]
    ])
    ca1 = CA(board=board)
    ca1.print()
    ca1.draw()
    
    ca1.update()
    ca1.print()
    ca1.draw()
    
def test_1_2():
    board = np.array([
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1]
    ])
    ca1 = CAv1(board=board, death_threshold=9, life_threshold=0)
    ca1.print()
    ca1.draw()
    
    ca1.update()
    ca1.print()
    ca1.draw()

def test_1_3():
    board = np.array([
        [1,0,0,0],
        [1,0,1,1],
        [0,0,0,0],
        [1,0,0,1]
    ])
    ca1 = CAv1(board=board, death_threshold=0, life_threshold=9)
    ca1.print()
    ca1.draw()
    
    ca1.update()
    ca1.print()
    ca1.draw()

def test_2():
    ca1 = CA(size=10, density=0.7)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()
        
def test_2_all_die():
    ca1 = CA(size=10, density=0.7, overpopulation=0, underpopulation=9, birth=10)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()
        
def test_2_all_die2():
    ca1 = CAv1(size=10, density=0.7, death_threshold=0, life_threshold=9)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()

def test_2_all_live():
    ca1 = CA(size=10, density=0.7, overpopulation=9, underpopulation=0, birth=3)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()
        
def test_2_all_live2():
    ca1 = CAv1(size=10, density=0.7, death_threshold=9, life_threshold=0)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()
        
def test_strategy1():
    ca1 = CA(rules_strategy=RulesStrategyBS, size=10, density=0.5, survival=[2,3], birth=[3])
    ca1.print()
    ca1.update()
    ca1.print()
        
def test_strategy2():
    ca1 = CA(rules_strategy=RulesStrategyUPOPB, size=10, density=0.7, overpopulation=3, underpopulation=2, birth=[3])
    ca1.print()
    ca1.update()
    ca1.print()

def test_strategy3():
    ca2 = CA(rules_strategy=RulesStrategyLD, size=10, density=0.7, life_threshold=9, death_threshold=0)
    ca2.print()
    ca2.update()
    ca2.print()
    
def test_CA_factory_1():
    ca1 = CAFactory().create_CA_BS(density=0.5, size=15)
    ca1.print()
    ca1.update()
    ca1.print()
    
def test_CA_factory_2():
    ca1 = CAFactory().create_CA_UPOPB(density=0.5, size=15)
    ca1.print()
    ca1.update()
    ca1.print()
    
def test_CA_factory_3():
    ca1 = CAFactory().create_CA_LB(density=0.5, size=15)
    ca1.print()
    ca1.update()
    ca1.print()
    
def test_draw_evolution1():
    np.random.seed(0)
    
    ca1 = CAFactory.create_CA_LB(
        life_threshold=2, 
        death_threshold=4, 
        size=10, 
        density=0.5, 
        iterations=3)
    
    ca1.print_evolution()
    ca1.draw_evolution()

def test_save_and_load1():
    ca1 = CAFactory.create_CA_LB(
        life_threshold=2, 
        death_threshold=4, 
        size=10, 
        density=0.5, 
        iterations=3)
    
    file = "./data/test/test"
    ca1.save_evolution(filename=file)
    
    ca1_loaded = np.load(file + ".npy")
    print(ca1_loaded.shape)
