import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from gym import Env, spaces
from typing import Tuple, TypeVar, Literal, List

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
PieceType = Literal[1, -1]
PlayerType = Literal[1, 2]
FOUR = 4
LostValue = -1
WonValue = 1

class Board:
    def __init__(self, rows: int = 6, cols: int = 7, win_at:int = 4) -> None:
        self.win_at = win_at
        self.shape = (rows, cols)
        self.reset()

    def reset(self) -> None:
        self.b_array = np.zeros(shape=self.shape, dtype=np.int8)

    @property
    def available_cols(self) -> List[int]:
        """ Returns list of columns that are not completely filled """
        return [idx for idx, col in enumerate(self.b_array.T) if np.count_nonzero(col) < len(col)]

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def board_filled(self) -> bool:
        """ Returns True if board is completely filled """
        return np.count_nonzero(self.b_array) >= self.b_array.size
    
    def add_piece(self, col:int, piece:PieceType) -> bool:
        """ Adds piece to given col, return False if addition not poosible """
        if col not in self.available_cols:
            return False
        # Determine addition index
        at = max(np.where(self.b_array.T[col] == 0)[0])
        self.b_array.T[col, at] = piece
        return True
    
    def won(self) -> bool:
        """ Returns true if win condition found """
        def found_pattern(b_array:np.ndarray) -> bool:
            """ Returns True if the required pattern found in 2D array """
            assert(b_array.ndim > 1), "2D array expected here"
            for rows in sliding_window_view(b_array, (1, self.win_at)):
                for window in rows:
                    if abs(np.sum(window)) == self.win_at:
                        return True
            return False

        def check_diagonal_arr(arr:np.ndarray) -> bool:
            """ Returns True if the diagonals contains a win condition. """
            for i in range(-arr.shape[0] + 1, arr.shape[1]):
                if abs(np.sum(np.diag(arr, k=i))) >= self.win_at:
                    return True
            return False
        return found_pattern(self.b_array) | found_pattern(self.b_array.T) | check_diagonal_arr(self.b_array) | check_diagonal_arr(np.fliplr(self.b_array))


class FourInRowEnv(Env):
    metadata = {"render_modes": None}

    def __init__(self, rows: int = 6, cols: int = 7) -> None:
        assert(rows > 4), "rows must > 4"
        assert(cols > 4), "cols must > 4"
        self.shape = (rows, cols)
        self.action_space = spaces.Discrete(cols * 2)
        self.board = Board(rows=self.rows, cols=self.cols, win_at=FOUR)
        self.reset()

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def rows(self) -> int:
        return self.shape[0]
    
    @property
    def state(self) -> np.ndarray:
        return self.board.b_array
    
    def reset(self) -> None:
        self.done = False
        self.board.reset()

    def col_to_action(self, col:int, player:PlayerType) -> ActType:
        """ Helper function to convert player column to action """
        assert (col < self.cols), "wrong move(col) recieved"
        assert (player in [1,2]), "wrong player"
        return col + self.cols if player == 2 else 0

    def add_piece(self, action: ActType) -> bool:
        """Adds a piece to col, value added depends on the col value
        Reutns True if addition was successful, else return False"""
        assert self.action_space.contains(action), "Action not in action_space!"
        piece = 1 if action >= self.cols else -1
        col = action % self.cols
        return self.board.add_piece(col=col, piece=piece)

    def has_won(self) -> bool:
        """Returns true if someone won!"""
        return self.board.won()

    def is_board_full(self) -> bool:
        """Returns true if the game has ended and no more places are left to add pieces"""
        return self.board.board_filled

    def step(self, action: ActType) -> Tuple[ObsType, int, bool, bool, dict]:
        """Returns (observations, reward, done)"""
        assert not self.done, "Game already finished"
        assert self.action_space.contains(action), "Action not part of action_space"
        reward, info = 0, None

        # If action is impossible return negative reward!
        if not self.add_piece(action):
            # We end the game if we get a wrong move from the agent!
            reward = LostValue
            self.done = True
        elif self.has_won():
            # Check if addition won the game
            reward = WonValue
            self.done = True
        else:
            self.done = self.is_board_full()
        return self.state, reward, self.done, info
