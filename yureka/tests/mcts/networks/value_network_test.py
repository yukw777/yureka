import torch
import chess

from yureka.mcts.networks import ValueNetwork
from unittest.mock import MagicMock


def test_get_value():
    t = torch.zeros(1, 1)
    mock_model = MagicMock(return_value=t)
    vn = ValueNetwork(mock_model, cuda=False)
    assert vn.get_value(chess.Board(), chess.WHITE) == 0
