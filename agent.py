import random
from abc import ABC, abstractmethod
from fir_env import ActType, Board, PlayerType

class FourRowAgent(ABC):
    @abstractmethod
    def get_action(self, board:Board) -> ActType:
        raise NotImplemented

class RandomPlayer(FourRowAgent):
    def get_action(self, board: Board) -> ActType:
        available_cols = board.available_cols
        action = available_cols[random.randrange(len(available_cols))]
        return action

