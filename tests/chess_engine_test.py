import torch
import chess
from chess_engine import ChessEngine
from unittest.mock import MagicMock
from torch.autograd import Variable


def test_get_move():
    test_cases = [
        {
            'expected_move_index': 0,
            'expected_white_move': chess.Move.from_uci('a1a2'),
            'expected_black_move': chess.Move.from_uci('h8h7'),
        },
        {
            'expected_move_index': 100,
            'expected_white_move': chess.Move.from_uci('e5f6'),
            'expected_black_move': chess.Move.from_uci('d4c3'),
        },
    ]
    for tc in test_cases:
        t = torch.randn(4672, 1)
        t[tc['expected_move_index']] = t.max() + 1
        mock_model = MagicMock(return_value=Variable(t))
        white_board = chess.Board()
        black_board = chess.Board()
        black_board.push(chess.Move.from_uci("g1f3"))
        e = ChessEngine(model=mock_model, cuda=False)
        assert tc['expected_white_move'] == e.get_move(white_board)
        assert tc['expected_black_move'] == e.get_move(black_board)
