from __future__ import annotations
from abc import ABC, abstractmethod

class RulesStrategy(ABC):
    
    @abstractmethod
    def update_cell(self, board, cell, i, j):
        pass

class RulesStrategyBS(RulesStrategy):
    
    def update_cell(self, board, cell, i, j):
        alive = cell == 1
        neighbours = self.get_neighbours(i, j)
        if alive and neighbours in self.S:
            board[i][j] = 1
        elif not alive and neighbours in self.B:
            board[i][j] = 1
        else:
            board[i][j] = 0


class RulesStrategyBISI(RulesStrategy):
    
    def update_cell(self, board, cell, i, j):
        alive = cell == 1
        neighbours = self.get_neighbours(i, j)
        bmin = self.B[0]
        bmax = self.B[1]
        smin = self.S[0]
        smax = self.S[1]
        if alive and neighbours >= smin and neighbours <= smax:
            board[i][j] = 1
        elif not alive and neighbours >= bmin and neighbours <= bmax:
            board[i][j] = 1
        else:
            board[i][j] = 0


class RulesStrategyBTST(RulesStrategy):
    
    def update_cell(self, board, cell, i, j):
        alive = cell == 1
        neighbours = self.get_neighbours(i, j)
        if alive and neighbours >= self.S:
            board[i][j] = 1
        elif not alive and neighbours >= self.B:
            board[i][j] = 1
        else:
            board[i][j] = 0
