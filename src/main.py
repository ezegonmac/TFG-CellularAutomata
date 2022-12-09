from tests.manual_tests import *
from datasets.datasets import *

if __name__ == '__main__':

    np.random.seed(0)
    
    ca1 = CAFactory.create_CA_LB(
        life_threshold=2, 
        death_threshold=4, 
        size=10, 
        density=0.5,
        iterations=3)
    
    ca1.draw_evolution()
