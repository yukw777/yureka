import random
import torch

from ...learn.data.move_translator import (
    TOTAL_MOVES,
    translate_to_engine_move,
    get_engine_move_index,
)

from .value_network import ValueNetwork
from .policy_network import PolicyNetwork


__all__ = ['ValueNetwork', 'PolicyNetwork']


class ZeroValue():
    def get_value(self, board, color):
        return 0


class RandomPolicy():
    def get_move(self, board, sample=True):
        return random.choice(list(board.legal_moves))

    def get_probs(self, board):
        moves = list(board.legal_moves)
        prob = 1/len(moves)
        probs = torch.zeros(1, TOTAL_MOVES)
        indeces = []
        for move in moves:
            engine_move = translate_to_engine_move(move, board.turn)
            index = get_engine_move_index(engine_move)
            indeces.append(index)
        probs.index_fill_(1, torch.LongTensor(indeces), prob)
        return probs
