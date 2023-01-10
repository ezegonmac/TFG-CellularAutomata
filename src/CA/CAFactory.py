from CA.CA import CA
from CA.RulesStrategies import *

class CAFactory():
    """
    Factory for creating CA objects with different rule strategy.

    Returns:
        CA: CA object with the specified rule strategy.
    """

    @staticmethod
    def create_CA_BS(B=[2,3], S=[3], size=None, density=None, board=None, iterations=None):
        """CA with B/S rule strategy.

        Args:
            B (list, optional): (Birth) List of live neighbors values that a dead cell has to become alive. Defaults to [2,3].
            S (list, optional): (Survival) List of live neighbors values that an alive cell has to have stay alive. Defaults to [3].
            
            iterations (int, optional): number of iterations to be conmputed. Defaults to 10.
            
            If you want the board to be generated automatically: 
            size (int, optional): width and height of the board. Defaults to None.
            density (int, optional): initial configuration average density. Defaults to None.
            
            If you want to specify the board:
            board (np.array, optional): specify initial configuration board. Defaults to None.

        Returns:
            CA: CA object with the specified B/S rule strategy.
        """
        return CA(B=B, S=S, rules_strategy=RulesStrategyBS, size=size, density=density, board=board, iterations=iterations)

    @staticmethod
    def create_CA_BISI(B=(2,4), S=(7,8), size=None, density=None, board=None, iterations=None):
        """CA with BI/SI rule strategy.

        Args:
            B (tuple, optional): (Birth) Tuple (Bl, Bt) defining the interval of live neighbors that a dead cell has to have stay become alive. Defaults to (2,4).
            S (tuple, optional): (Survival) Tuple (Sl, St) defining the interval of live neighbors that an alive cell has to have stay alive. Defaults to (7,8).

            iterations (int, optional): number of iterations to be conmputed. Defaults to 10.
            
            If you want the board to be generated automatically: 
            size (int, optional): width and height of the board. Defaults to None.
            density (int, optional): initial configuration average density. Defaults to None.
            
            If you want to specify the board:
            board (np.array, optional): specify initial configuration board. Defaults to None.

        Returns:
            CA: CA object with the specified BI/SI rule strategy.
        """
        return CA(S=S, B=B, rules_strategy=RulesStrategyBISI, size=size, density=density, board=board, iterations=iterations)

    @staticmethod
    def create_CA_BTST(B=7, S=4, size=None, density=None, board=None, iterations=None):
        """CA with BT/ST rule strategy.

        Args:
            B (int, optional): (Birth) Number of live neighbors from which a dead cell become alive. Defaults to 7.
            S (int, optional): (Survival) Number of live neighbors from which an alive cell stay alive. Defaults to 4.
    
            iterations (int, optional): number of iterations to be conmputed. Defaults to 10.
            
            If you want the board to be generated automatically: 
            size (int, optional): width and height of the board. Defaults to None.
            density (int, optional): initial configuration average density. Defaults to None.
            
            If you want to specify the board:
            board (np.array, optional): specify initial configuration board. Defaults to None.

        Returns:
            CA: CA object with the specified BT/ST rule strategy.
        """
        return CA(B=B, S=S, rules_strategy=RulesStrategyBTST, size=size, density=density, board=board, iterations=iterations)
