import random
import torch

from torch.autograd import Variable

from ...move_translator import (
    TOTAL_MOVES,
    translate_to_engine_move,
    get_engine_move_index,
)

from .value_network import ValueNetwork
from .policy_network import PolicyNetwork


__all__ = ['ValueNetwork', 'PolicyNetwork']


class ZeroValue():
    def get_value(self, board):
        return 0


class RandomPolicy():
    def get_move(self, board, sample=True):
        return random.choice(list(board.legal_moves))

    def get_probs(self, board):
        moves = list(board.legal_moves)
        prob = 1/len(moves)
        probs = Variable(torch.zeros(1, TOTAL_MOVES))
        indexes = []
        for move in moves:
            engine_move = translate_to_engine_move(move, board.turn)
            index = get_engine_move_index(engine_move)
            indexes.append(index)
        probs.index_fill_(1, Variable(torch.LongTensor(indexes)), prob)
        return probs
