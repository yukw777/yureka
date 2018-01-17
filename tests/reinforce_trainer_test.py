import chess
import torch
import unittest.mock as mock
from reinforce_trainer import ReinforceTrainer
from torch.autograd import Variable


def test_self_play():
    # use fool's mate to test (white loses)
    white = mock.MagicMock()
    white.get_move.side_effect = [
        (chess.Move.from_uci('f2f3'), Variable(torch.Tensor([0.5]).log())),
        (chess.Move.from_uci('g2g4'), Variable(torch.Tensor([0.7]).log())),
        chess.Move.from_uci('f2f3'),
        chess.Move.from_uci('g2g4'),
        chess.Move.from_uci('f6f7'),
        chess.Move.from_uci('e6f6'),
    ]
    black = mock.MagicMock()
    black.get_move.side_effect = [
        chess.Move.from_uci('e7e5'),
        chess.Move.from_uci('d8h4'),
        (chess.Move.from_uci('e7e5'), Variable(torch.Tensor([0.5]).log())),
        (chess.Move.from_uci('d8h4'), Variable(torch.Tensor([0.7]).log())),
        (chess.Move.from_uci('e8f8'), Variable(torch.Tensor([0.7]).log())),
    ]

    # trainee loses
    t = ReinforceTrainer('ChessEngine.v0', 'pool_path', None)
    reward, policy_loss = t.self_play(white, black, chess.WHITE)
    assert reward == -1
    assert type(policy_loss) == Variable

    # trainee wins
    t = ReinforceTrainer('ChessEngine.v0', 'pool_path', None)
    reward, policy_loss = t.self_play(black, white, chess.BLACK)
    assert reward == 1
    assert type(policy_loss) == Variable

    # tie game
    mock_board = mock.MagicMock()
    mock_board.return_value = chess.Board(fen='4k3/8/4KP2/8/8/8/8/8 w - - 0 1')
    with mock.patch('chess.Board', mock_board):
        t = ReinforceTrainer('ChessEngine.v0', 'pool_path', None)
        reward, policy_loss = t.self_play(black, white, chess.BLACK)
        assert reward == 0
        assert type(policy_loss) == Variable
