from manual_tests import *

if __name__ == '__main__':
    
    # test_1()
    # test_1_2()
    # test_1_3()
    
    # test_2()
    # test_2_all_live()
    # test_2_all_die()
    
    # test_2_all_live2()
    # test_2_all_die2()

    # test_strategy1()
    # test_strategy2()
    # test_strategy3()
    
    # test_CA_factory_1()
    # test_CA_factory_2()
    # test_CA_factory_3()

    # test_draw_evolution1()

    ca1 = CAFactory.create_CA_LB(
        life_threshold=2, 
        death_threshold=4, 
        size=10, 
        density=0.5, 
        iterations=3)
    
    ca1.save_evolution(filename='test')