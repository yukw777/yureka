import torch
import math
import chess

from yureka.chess_engine import (
    ChessEngine,
    queen_promotion_if_possible,
)
from yureka.engine import UCIPolicyEngine
from yureka.engine.constants import DEFAULT_MODEL, DEFAULT_MODEL_FILE
from unittest.mock import MagicMock, patch
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
            with patch('yureka.chess_engine.torch.nn.functional.softmax',
                       return_value=Variable(t)):
                white_move, log_prob = e.get_move(white_board)
                assert tc['expected_white_move'] == white_move
                assert type(log_prob) == Variable

                black_move, log_prob = e.get_move(black_board)
                assert tc['expected_black_move'] == black_move
                assert type(log_prob) == Variable
        else:
            assert tc['expected_white_move'] == e.get_move(white_board)
            assert tc['expected_black_move'] == e.get_move(black_board)


def test_get_move_probs_zero():
    t = torch.zeros(1, 4672)
    mock_model = MagicMock(return_value=Variable(t))
    board = chess.Board()
    board.push(chess.Move.from_uci("g1f3"))
    e_train = ChessEngine(model=mock_model, cuda=False, train=True)
    e_test = ChessEngine(model=mock_model, cuda=False, train=False)

    move, log_prob = e_train.get_move(board)
    assert round(log_prob.data[0], 6) == round(math.log(1/20), 6)
    assert move in board.legal_moves
    move = e_test.get_move(board)
    assert move in board.legal_moves


def test_queen_promotion():
    test_cases = [
        {
            'board': chess.Board(fen='8/4P3/8/8/8/8/8/8 w - - 0 1'),
            'move': chess.Move.from_uci('e7e8'),
            'expected_move': chess.Move.from_uci('e7e8q'),
        },
        {
            'board': chess.Board(fen='8/8/8/8/8/8/4p3/8 b - - 0 1'),
            'move': chess.Move.from_uci('e2e1'),
            'expected_move': chess.Move.from_uci('e2e1q'),
        },
        {
            'board': chess.Board(fen='8/8/8/8/8/8/4p3/8 b - - 0 1'),
            'move': chess.Move.from_uci('e2e1r'),
            'expected_move': chess.Move.from_uci('e2e1r'),
        },
        {
            'board': chess.Board(fen='8/4R3/8/8/8/8/8/8 w - - 0 1'),
            'move': chess.Move.from_uci('e7e8'),
            'expected_move': chess.Move.from_uci('e7e8'),
        },
        {
            'board': chess.Board(fen='8/8/8/8/8/8/4r3/8 b - - 0 1'),
            'move': chess.Move.from_uci('e2e1'),
            'expected_move': chess.Move.from_uci('e2e1'),
        },
        {
            'board': chess.Board(fen='8/8/8/8/3P4/8/8/8 w - - 0 1'),
            'move': chess.Move.from_uci('d4d5'),
            'expected_move': chess.Move.from_uci('d4d5'),
        },
        {
            'board': chess.Board(fen='8/8/8/8/3p4/8/8/8 b - - 0 1'),
            'move': chess.Move.from_uci('d4d3'),
            'expected_move': chess.Move.from_uci('d4d3'),
        },
    ]

    for tc in test_cases:
        converted = queen_promotion_if_possible(tc['board'], tc['move'])
        assert tc['expected_move'] == converted


def uci_setoption_test():
    test_cases = [
        {
            'args': 'setoption name Model File value file_name',
            'attr': 'model_file',
            'value': 'file_name',
        },
        {
            'args': 'setoption name Model Name value model_name',
            'attr': 'model_name',
            'value': 'model_name',
        },
        {
            'args': 'setoption name Cuda Device value 2',
            'attr': 'cuda_device',
            'value': 2,
        },
        {
            'args': 'setoption name Unknown value 2',
            'attr': 'cuda_device',
            'value': None,
        },
        {
            'args': 'setoption name Unknown value 2',
            'attr': 'model_file',
            'value': DEFAULT_MODEL_FILE,
        },
        {
            'args': 'setoption name Unknown value 2',
            'attr': 'model_name',
            'value': DEFAULT_MODEL,
        },
    ]
    for tc in test_cases:
        uci = UCIPolicyEngine()
        uci.setoption(tc['args'])
        assert getattr(uci, tc['attr']) == tc['value']


def test_uci_position():
    test_cases = [
        {
            'args': 'startpos moves e2e4 c7c5 b1c3 b8c6 g1f3 g7g6',
            'board': chess.Board(),
            'moves': ['e2e4', 'c7c5', 'b1c3', 'b8c6', 'g1f3', 'g7g6'],
        },
        {
            'args': 'startpos',
            'board': chess.Board(),
            'moves': [],
        },
        {
            'args': 'fen r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R'
                    ' b KQkq - 3 3',
            'board': chess.Board(fen='r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/'
                                 'PPPP1PPP/R1BQKB1R b KQkq - 3 3'),
            'moves': [],
        },
        {
            'args': 'fen r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R'
                    ' b KQkq - 3 3 moves a7a5 a2a4',
            'board': chess.Board(fen='r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/'
                                 'PPPP1PPP/R1BQKB1R b KQkq - 3 3'),
            'moves': ['a7a5', 'a2a4'],
        },
    ]

    for tc in test_cases:
        uci = UCIPolicyEngine()
        uci.position(tc['args'])
        for m in tc['moves']:
            tc['board'].push_uci(m)
        assert uci.board == tc['board']
