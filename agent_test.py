from agent import RandomPlayer
from fir_env import Board


class Test_random_agent:
    def test_actions(self):
        r, c = 2, 2
        board = Board(rows=r, cols=c, win_at=2)
        agent = RandomPlayer()
        for _ in range(r*c):
            board.add_piece(agent.get_action(board), -1)
        assert(board.available_cols == []), "Board must be fulled now!"

    

    