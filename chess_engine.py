import attr
import collections
import state_generator
import chess_dataset
import torch
from torch.autograd import Variable
from move_translator import (
    translate_to_engine_move,
    translate_from_engine_move,
    get_engine_move_from_index,
    get_engine_move_index,
)


@attr.s
class ChessEngine():
    model = attr.ib()
    cuda = attr.ib(default=True)
    transpositions = attr.ib(default=collections.Counter())

    def __attrs_post_init__(self):
        self.cuda = self.cuda and torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()
        self.model.eval()
        self.transpositions = collections.Counter()

    def get_move(self, board):
        board_data = state_generator.get_board_data(board, self.transpositions)
        tensor = chess_dataset.get_tensor_from_row(board_data)
        tensor = tensor.view(1, *tensor.shape)
        volatile = not self.model.training
        if self.cuda:
            inputs = Variable(tensor.cuda(), volatile=volatile)
        else:
            inputs = Variable(tensor, volatile=volatile)
        outputs = self.model(inputs)
        outputs = outputs.view(outputs.shape[0], -1)
        legal_outputs = filter_illegal_moves(board, outputs)
        _, move_index = legal_outputs.max(1)
        engine_move = get_engine_move_from_index(move_index.data[0])
        return translate_from_engine_move(engine_move, board.turn)


def filter_illegal_moves(board, probs):
    filtered = Variable(torch.zeros(probs.shape))
    for move in board.legal_moves:
        engine_move = translate_to_engine_move(move, board.turn)
        index = get_engine_move_index(engine_move)
        filtered.data[0, index] = probs.data[0, index]
    probs.set_(source=filtered)

    return probs
