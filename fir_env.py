import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from gym import Env, spaces
from typing import Tuple, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
Four = 4
LostValue = -1
WonValue = 1


class FourInRowEnv(Env):
    metadata = {"render_modes": None}

    def __init__(self, rows: int = 6, cols: int = 7) -> None:
        self.shape = (rows, cols)
        self.action_space = spaces.Discrete(cols * 2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.shape, dtype=np.int8
        )
        self.reset()

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def rows(self) -> int:
        return self.shape[0]

    def reset(
        self,
    ) -> None:
        self.done = False
        self.state = np.zeros(shape=self.shape, dtype=np.int8)

    def add_piece(self, action: ActType) -> bool:
        """Adds a piece to col, value added depends on the col value
        Reutns True if addition was successful, else return False"""
        assert self.action_space.contains(action), "Action not in action_space!"
        val = 1 if action >= self.cols else -1
        col = action % self.cols
        for row in range(self.rows - 1, -1, -1):
            if self.state[row, col] == 0:
                self.state[row, col] = val
                return True
        return False

    def has_won(self) -> bool:
        """Returns true if someone won!"""

        def slide_window_check(state: ObsType) -> bool:
            """Slide a window side of 4 along the row/s"""
            for rows in sliding_window_view(state, (1, Four)):
                for window in rows:
                    # print(np.sum(window))
                    if abs(np.sum(window)) == Four:
                        return True
            return False

        def check_diagnols(state: ObsType) -> bool:
            """Determines a numpy array of diagonals and slide the checks"""
            ncols = state.shape[1]
            # Create an array of diagonal where each row is a diagonal
            diag_array1 = np.array(
                [
                    np.resize(
                        np.diag(state, k=i),
                        ncols,
                    )
                    for i in range(-ncols + 1 + Four, ncols + 1 - Four)
                ]
            )
            # Flip the array and determine the diagonals again
            h_flipped = np.fliplr(state)
            diag_array2 = np.array(
                [
                    np.resize(
                        np.diag(h_flipped, k=i),
                        ncols,
                    )
                    for i in range(-ncols + 1 + Four, ncols + 1 - Four)
                ]
            )
            return slide_window_check(np.vstack((diag_array1, diag_array2)))

        # Check all 4 way possible winning senarios
        four_in_rows = slide_window_check(self.state)
        four_in_cols = slide_window_check(self.state.T)
        four_in_diag = check_diagnols(self.state)

        return four_in_rows | four_in_cols | four_in_diag

    def is_board_full(self) -> bool:
        """Returns true if the game has ended and no more places are left to add pieces"""
        return self.state.all()

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
