import torch
import numpy as np
from yureka.learn.data.chess_dataset import ChessDataset
from yureka.learn.data.board_data import BOARD_SIZE


def test_chess_dataset():
    dataset = ChessDataset('yureka/tests/test.engine.csv')
    assert len(dataset) == 4

    test_cases = [
        {
            'data': dataset[0][0],
            'move': dataset[0][1],
            'value': dataset[0][2],
            'black_p': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'black_n': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'black_b': np.array([[0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'black_r': np.array([[0, 0, 1, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'black_q': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'black_k': np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_p': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_n': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_b': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0]]),
            'white_r': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 1]]),
            'white_q': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_k': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0]]),
            'rep_2': 0,
            'rep_3': 0,
            'turn': 0,  # black
            'move_count': 15,
            'b_k_castling': 0,
            'b_q_castling': 0,
            'w_k_castling': 0,
            'w_q_castling': 0,
            'no_progress': 0,
            'expected_move': 642,
            'expected_value': -1,
        },
        {
            'data': dataset[3][0],
            'move': dataset[3][1],
            'value': dataset[3][2],
            'white_p': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_n': np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_b': np.array([[0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_r': np.array([[1, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_q': np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'white_k': np.array([[0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'black_p': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'black_n': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]]),
            'black_b': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0]]),
            'black_r': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 1]]),
            'black_q': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0]]),
            'black_k': np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0]]),
            'rep_2': 0,
            'rep_3': 0,
            'turn': 1,  # white
            'move_count': 6,
            'b_k_castling': 1,
            'b_q_castling': 1,
            'w_k_castling': 0,
            'w_q_castling': 0,
            'no_progress': 0,
            'expected_move': 817,
            'expected_value': 0,
        },
    ]

    for tc in test_cases:
        # White pieces = 6 * 8
        # Black pieces = 6 * 8
        # Constnat plane filled with one = 1
        # Repetitions = 2 * 8
        # Color = 1
        # Move count = 1
        # White castling = 2
        # Black castling = 2
        # No-progress count = 1
        # Constnat plane filled with zeroe = 1
        # Total = 121
        assert tc['data'].shape == (121, 8, 8)

        assert tc['data'][0].equal(torch.from_numpy(tc['black_p']).float())
        assert tc['data'][1].equal(torch.from_numpy(tc['black_n']).float())
        assert tc['data'][2].equal(torch.from_numpy(tc['black_b']).float())
        assert tc['data'][3].equal(torch.from_numpy(tc['black_r']).float())
        assert tc['data'][4].equal(torch.from_numpy(tc['black_q']).float())
        assert tc['data'][5].equal(torch.from_numpy(tc['black_k']).float())

        assert tc['data'][48].equal(torch.from_numpy(tc['white_p']).float())
        assert tc['data'][49].equal(torch.from_numpy(tc['white_n']).float())
        assert tc['data'][50].equal(torch.from_numpy(tc['white_b']).float())
        assert tc['data'][51].equal(torch.from_numpy(tc['white_r']).float())
        assert tc['data'][52].equal(torch.from_numpy(tc['white_q']).float())
        assert tc['data'][53].equal(torch.from_numpy(tc['white_k']).float())

        assert tc['data'][96].equal(
            torch.from_numpy(np.full(BOARD_SIZE, 1)).float())

        assert tc['data'][97].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['rep_2'])).float())
        assert tc['data'][105].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['rep_3'])).float())
        assert tc['data'][113].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['turn'])).float())
        assert tc['data'][114].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['move_count'])).float())
        assert tc['data'][115].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['b_k_castling'])).float())
        assert tc['data'][116].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['b_q_castling'])).float())
        assert tc['data'][117].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['w_k_castling'])).float())
        assert tc['data'][118].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['w_q_castling'])).float())
        assert tc['data'][119].equal(
            torch.from_numpy(np.full(BOARD_SIZE, tc['no_progress'])).float())

        assert tc['data'][120].equal(
            torch.from_numpy(np.full(BOARD_SIZE, 0)).float())

        assert tc['move'] == tc['expected_move']
        assert tc['value'].equal(torch.Tensor([float(tc['expected_value'])]))
