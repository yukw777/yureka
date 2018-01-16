import torch
import chess
from chess_engine import ChessEngine
from unittest.mock import MagicMock
from torch.autograd import Variable


def test_get_move():
    test_cases = [
        {
            'expected_move_index': 9,
            'expected_white_move': chess.Move.from_uci('b2b3'),
            'expected_black_move': chess.Move.from_uci('g7g6'),
            'train': True,
        },
        {
            'expected_move_index': 520,
            'expected_white_move': chess.Move.from_uci('a2a4'),
            'expected_black_move': chess.Move.from_uci('h7h5'),
            'train': True,
        },
        {
            'expected_move_index': 9,
            'expected_white_move': chess.Move.from_uci('b2b3'),
            'expected_black_move': chess.Move.from_uci('g7g6'),
            'train': False,
        },
        {
            'expected_move_index': 520,
            'expected_white_move': chess.Move.from_uci('a2a4'),
            'expected_black_move': chess.Move.from_uci('h7h5'),
            'train': False,
        },
    ]
    for tc in test_cases:
        t = torch.zeros(1, 4672)
        t[0, tc['expected_move_index']] = t.max() + 1
        mock_model = MagicMock(return_value=Variable(t))
        white_board = chess.Board()
        black_board = chess.Board()
        black_board.push(chess.Move.from_uci("g1f3"))
        e = ChessEngine(model=mock_model, cuda=False, train=tc['train'])
        if tc['train']:
            white_move, log_prob = e.get_move(white_board)
            assert tc['expected_white_move'] == white_move
            assert type(log_prob) == Variable

            black_move, log_prob = e.get_move(black_board)
            assert tc['expected_black_move'] == black_move
            assert type(log_prob) == Variable
        else:
            assert tc['expected_white_move'] == e.get_move(white_board)
            assert tc['expected_black_move'] == e.get_move(black_board)
