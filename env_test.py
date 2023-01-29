import pytest
import numpy as np
from fir_env import FourInRowEnv, LostValue, WonValue


def test_env_step():
    env = FourInRowEnv()
    env.step(action=1)

def test_action_space():
    env = FourInRowEnv()
    action_space = env.action_space
    for i in range(env.shape[1] * 2):
        assert(action_space.contains(i))

    with pytest.raises(AssertionError):
        assert(action_space.contains(env.shape[1] * 2 + 1))

def test_add_piece():
    """ Test addition of piece and return value """
    env = FourInRowEnv()
    assert(env.rows % 2 == 0), "This test requires the rows to be even"
    for _ in range(int(env.rows / 2)):
        assert(env.add_piece(0) == True)
        assert(env.add_piece(env.cols) == True)
    
    assert(env.add_piece(0) == False), "Cannot insert anymore"
    assert(env.add_piece(env.cols) == False), "Cannot insert anymore"
    print(env.state)

def test_reset():
    """ Test the resets """
    env = FourInRowEnv()
    env.add_piece(0)
    assert(env.state.any()) , "There has to be a non-zero element"
    env.reset()
    assert(not env.state.any()), "After reset all values must be zero"

def test_add_piece_assertion():
    """ Test assertion if wrong col is passed! """
    env = FourInRowEnv()
    with pytest.raises(AssertionError):
        env.add_piece(-1)

    with pytest.raises(AssertionError):
        env.add_piece(env.cols * 2)

def test_has_won():
    env = FourInRowEnv()
    assert(env.has_won() == False), "When Board is empty no one won"

    # Check in cols
    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == False)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == True)

    env.state = np.array([
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == True)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1, -1,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == False)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0, -1, -1, -1,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == False)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0, -1,  1, -1,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0, -1, -1, -1,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == False)


    # Check rows win
    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 1,  1,  1,  0,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == False)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 1,  1,  1,  0, -1,  1,  1,]
        ], dtype=np.int8)
    assert(env.has_won() == False)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 1,  1,  1,  1,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == True)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 1,  1, -1,  1,  1,  1,  1,]
        ], dtype=np.int8)
    assert(env.has_won() == True)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 1,  1,  1, -1, -1,  1,  1,],
            [ 1,  1,  1, -1, -1,  1,  1,],
            [ 1,  1,  1, -1, -1,  1,  1,]
        ], dtype=np.int8)
    assert(env.has_won() == False)

    # Check diagnols
    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0, -1,  1,  0,  0,],
            [ 0,  0,  0, -1, -1,  1,  0,],
            [ 1,  0,  0, -1, -1, -1,  1,]
        ], dtype=np.int8)
    assert(env.has_won() == False)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  0, -1,  1,  0,  0,],
            [ 0,  0,  0, -1, -1,  1,  0,],
            [ 1,  0,  0, -1, -1, -1,  1,]
        ], dtype=np.int8)
    assert(env.has_won() == True)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  1, -1,  0,  0,  0,],
            [ 0,  1,  0, -1,  0,  0,  0,],
            [ 1, -1, -1, -1,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == True)

    env.state = np.array([
            [ 0,  0,  0,  0,  1,  0,  0,],
            [ 0,  0,  0,  0,  0,  1,  0,],
            [ 0,  0,  0,  0,  0,  0,  1,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == False)
    
    env.state = np.array([
            [ 0,  0,  0,  0,  1,  0,  0,],
            [ 0,  0,  0,  0,  0,  1,  0,],
            [-1,  0,  0,  0,  0,  0,  1,],
            [ 0, -1,  0,  0,  0,  0,  0,],
            [ 0,  0, -1,  0,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == True)
    
    env.state = np.array([
            [ 0,  0,  0,  0, -1,  0,  1,],
            [ 0,  0,  0,  0,  0,  1,  0,],
            [ 0,  0,  0,  0,  1,  0, -1,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == True)

    env.state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  1,  0,],
            [ 0,  0,  0,  0,  1,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  1,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,]
        ], dtype=np.int8)
    assert(env.has_won() == True)


def test_is_board_full():
    env = FourInRowEnv()
    env.state = np.array([
            [ 1,  1, -1,  1,  1,  1,  0,],
            [ 1,  1, -1,  1,  1,  1,  1,],
            [ 1,  1, -1,  1,  1,  1,  1,],
            [ 1,  1, -1,  1,  1,  1,  1,],
            [ 1,  1,  1, -1, -1, -1,  1,],
            [ 1,  1,  1, -1, -1, -1,  1,]
        ], dtype=np.int8)
    assert(env.is_board_full() == False), "Game was not ended here"

    env.state = np.array([
            [ 1,  1, -1,  1,  1,  1,  1,],
            [ 1,  1, -1,  1,  1,  1,  1,],
            [ 1,  1, -1,  1,  1,  1,  1,],
            [ 1,  1, -1,  1,  1,  1,  1,],
            [ 1,  1,  1, -1, -1, -1,  1,],
            [ 1,  1,  1, -1, -1, -1,  1,]
        ], dtype=np.int8)
    assert(env.is_board_full() == True), "Game ended here"

def test_step():
    env = FourInRowEnv()
    # test the step function given a senario
    state, reward, done, info = env.step(action=0)
    ref_state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [-1,  0,  0,  0,  0,  0,  0,]
        ], dtype=np.int8)
    assert(np.array_equal(state, ref_state)), "New state not same as ref"
    assert((reward, done, info) == (0, False, None)), "Checking other parameters of the state"

    # Update the senario
    env.step(action=0)
    env.step(action=0)
    state, reward, done, info = env.step(action=8)
    ref_state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [-1,  0,  0,  0,  0,  0,  0,],
            [-1,  0,  0,  0,  0,  0,  0,],
            [-1,  1,  0,  0,  0,  0,  0,]
        ], dtype=np.int8)
    assert(np.array_equal(state, ref_state)), "New state not same as ref"
    assert((reward, done, info) == (0, False, None)), "Checking other parameters of the state"

    # Create a winning senario
    state, reward, done, info = env.step(action=0)
    ref_state = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,],
            [ 0,  0,  0,  0,  0,  0,  0,],
            [-1,  0,  0,  0,  0,  0,  0,],
            [-1,  0,  0,  0,  0,  0,  0,],
            [-1,  0,  0,  0,  0,  0,  0,],
            [-1,  1,  0,  0,  0,  0,  0,]
        ], dtype=np.int8)
    assert(np.array_equal(state, ref_state)), "New state not same as ref"
    assert((reward, done, info) == (WonValue, True, None)), "Checking other parameters of the state"

    # Check a wrong move
    env.reset()
    env.state = np.array([
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,],
            [ 0,  0,  0,  1,  0,  0,  0,],
            [ 0,  0,  0, -1,  0,  0,  0,]
        ], dtype=np.int8)
    _, reward, done, _ = env.step(action=3)
    assert(reward == LostValue), "Worng reward recieved"
    assert(done == True), "Game must be done"