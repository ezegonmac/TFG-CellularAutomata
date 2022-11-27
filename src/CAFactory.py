from CA import CA
from RulesStrategies import *
import numpy as np

class CAFactory():
    """
    Factory for creating CA objects with different rule strategy.

    Returns:
        CA: CA object with the specified rule strategy.
    """

    @staticmethod
    def create_CA_BS(birth=[2,3], survival=[3], size=None, density=None, board=None, iterations=None):
        """CA with B/S rule strategy.

        Args:
            birth (list, optional): List of of all the numbers of live neighbors that cause a dead cell to come alive (be born). Defaults to [2,3].
            survival (list, optional): List of all the numbers of live neighbors that cause a live cell to remain alive (survive). Defaults to [3].
            size (int, optional): Size of the square board to be generated. Defaults to None.
            density (float, optional): Density of the square board to be generated. Defaults to None.
            board (np.Array, optional): Board given by input. Defaults to None.
            iterations (int, optional): Number of iterations. Defaults to None.

        Returns:
            CA: CA object with the specified B/S rule strategy.
        """
        return CA(birth=birth, survival=survival, rules_strategy=RulesStrategyBS, size=size, density=density, board=board, iterations=iterations)

    @staticmethod
    def create_CA_UPOPB(underpopulation=2, overpopulation=3, birth=[3], size=None, density=None, board=None, iterations=None):
        """CA with UPOPB rule strategy.

        Args:
            underpopulation (int, optional): Number of live neighbors up to which a dead cell die (becouse of underpopulation). Defaults to 2.
            overpopulation (int, optional): Number of live neighbors from which a live cell die (becouse of overpopulation). Defaults to 3.
            birth (list, optional): List of all the numbers of life neighbors that cause a dead cell to become alive (be born). Defaults to [3].
            size (int, optional): Size of the square board to be generated. Defaults to None.
            density (float, optional): Density of the square board to be generated. Defaults to None.
            board (np.Array, optional): Board given by input. Defaults to None.
            iterations (int, optional): Number of iterations. Defaults to None.

        Returns:
            CA: CA object with the specified UPOPB rule strategy.
        """
        return CA(overpopulation=overpopulation, underpopulation=underpopulation, birth=birth, rules_strategy=RulesStrategyUPOPB, size=size, density=density, board=board, iterations=iterations)

    @staticmethod
    def create_CA_LB(life_threshold=7, death_threshold=4, size=None, density=None, board=None, iterations=None):
        """CA with LB rule strategy.

        Args:
            life_threshold (int, optional): Number of live neighbors from which a dead cell become alive. Defaults to 7.
            death_threshold (int, optional): Number of live neighbors from which a live cell die. Defaults to 4.
            size (int, optional): Size of the square board to be generated. Defaults to None.
            density (float, optional): Density of the square board to be generated. Defaults to None.
            board (np.Array, optional): Board given by input. Defaults to None.
            iterations (int, optional): Number of iterations. Defaults to None.

        Returns:
            CA: CA object with the specified LB rule strategy.
        """
        return CA(life_threshold=life_threshold, death_threshold=death_threshold, rules_strategy=RulesStrategyLD, size=size, density=density, board=board, iterations=iterations)