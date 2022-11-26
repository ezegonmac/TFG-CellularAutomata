from __future__ import annotations
from abc import ABC, abstractmethod

class RulesStrategy(ABC):
    
    @abstractmethod
    def update_cell(self, board, cell, i, j):
        pass

class RulesStrategyUPOPB(RulesStrategy):
    
    def update_cell(self, board, cell, i, j):
        alive = cell == 1
        neighbours = self.get_neighbours(i, j)
        if alive:
            if neighbours < self.underpopulation or neighbours > self.overpopulation:
                board[i][j] = 0
            else:
                board[i][j] = cell
        else:
            if neighbours == self.birth:
                board[i][j] = 1

class RulesStrategyLD(RulesStrategy):
    
    def update_cell(self, board, cell, i, j):
        alive = cell == 1
        neighbours = self.get_neighbours(i, j)
        if alive and neighbours >= self.death_threshold:
            board[i][j] = 0
        elif not alive and neighbours >= self.life_threshold:
            board[i][j] = 1
        else:
            board[i][j] = cell