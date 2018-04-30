import attr
import torch
import random

from torch.autograd import Variable

from yureka.board_data import get_board_data
from yureka.move_translator import (
    TOTAL_MOVES,
    translate_to_engine_move,
    get_engine_move_index,
)
from yureka import chess_dataset


@attr.s
class ValueNetwork():
    network = attr.ib()
    cuda = attr.ib(default=True)
    cuda_device = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.network.eval()
        self.cuda = self.cuda and torch.cuda.is_available()
        if self.cuda:
            self.network.cuda(self.cuda_device)

    def get_value(self, board):
        board_data = get_board_data(board)
        tensor = chess_dataset.get_tensor_from_row(board_data)
        tensor = tensor.unsqueeze(0)
        value = self.network(Variable(tensor.cuda(), volatile=True))
        return value.squeeze().data[0]


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
