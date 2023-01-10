import numpy as np
import pandas as pd
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
    ca1 = CAv1(board=board, S=9, B=0)
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
    ca1 = CAv1(board=board, S=0, B=9)
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
    ca1 = CA(size=10, density=0.7, S=0, B=9, B=10)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()
        
def test_2_all_die2():
    ca1 = CAv1(size=10, density=0.7, S=0, B=9)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()

def test_2_all_live():
    ca1 = CA(size=10, density=0.7, S=9, B=0, B=3)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()
        
def test_2_all_live2():
    ca1 = CAv1(size=10, density=0.7, S=9, B=0)
    ca1.print()
    ca1.draw()
    
    iterations = 3
    for i in range(0, iterations):
        ca1.update()
        ca1.print()
        ca1.draw()
        
def test_strategy1():
    ca1 = CA(rules_strategy=RulesStrategyBS, size=10, density=0.5, S=[2,3], B=[3])
    ca1.print()
    ca1.update()
    ca1.print()
        
def test_strategy2():
    ca1 = CA(rules_strategy=RulesStrategyBISI, size=10, density=0.7, S=3, B=2, B=[3])
    ca1.print()
    ca1.update()
    ca1.print()

def test_strategy3():
    ca2 = CA(rules_strategy=RulesStrategyBTST, size=10, density=0.7, B=9, S=0)
    ca2.print()
    ca2.update()
    ca2.print()
    
def test_CA_factory_1():
    ca1 = CAFactory().create_CA_BS(density=0.5, size=15)
    ca1.print()
    ca1.update()
    ca1.print()
    
def test_CA_factory_2():
    ca1 = CAFactory().create_CA_BISI(density=0.5, size=15)
    ca1.print()
    ca1.update()
    ca1.print()
    
def test_CA_factory_3():
    ca1 = CAFactory().create_CA_BTST(density=0.5, size=15)
    ca1.print()
    ca1.update()
    ca1.print()
    
def test_draw_evolution1():
    np.random.seed(0)
    
    ca1 = CAFactory.create_CA_BTST(
        B=2, 
        S=4, 
        size=10, 
        density=0.5, 
        iterations=3)
    
    ca1.print_evolution()
    ca1.draw_evolution()

def test_save_and_load1():
    ca1 = CAFactory.create_CA_BTST(
        B=2, 
        S=4, 
        size=10, 
        density=0.5, 
        iterations=3)
    
    file = "./data/test/test"
    ca1.save_evolution(filename=file)
    
    ca1_loaded = np.load(file + ".npy")
    print(ca1_loaded.shape)
    
def test_load_dataset1_evolution_density():
    df = pd.read_csv('./data/dataset1/dataset.csv', converters={'evolution_density': pd.eval})
    print(type(df.iloc[0].evolution_density))
