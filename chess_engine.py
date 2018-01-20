import attr
import collections
import chess
import state_generator
import chess_dataset
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
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
    cuda_device = attr.ib(default=None)
    transpositions = attr.ib(default=collections.Counter())
    train = attr.ib(default=True)

    def __attrs_post_init__(self):
        self.cuda = self.cuda and torch.cuda.is_available()
        if self.cuda:
            self.model.cuda(self.cuda_device)
        if self.train:
            self.model.train()
        else:
            self.model.eval()
        self.transpositions = collections.Counter()

    def get_move(self, board):
        board_data = state_generator.get_board_data(board, self.transpositions)
        tensor = chess_dataset.get_tensor_from_row(board_data)
        tensor = tensor.view(1, *tensor.shape)
        volatile = not self.model.training
        if self.cuda:
            inputs = Variable(tensor.cuda(self.cuda_device), volatile=volatile)
        else:
            inputs = Variable(tensor, volatile=volatile)
        outputs = self.model(inputs)
        probs = outputs.view(outputs.shape[0], -1)
        cloned_probs = probs.clone()
        self.filter_illegal_moves(board, probs)
        if self.train:
            m = Categorical(probs)
            move_index = m.sample()
            probs.set_(source=cloned_probs)
        else:
            _, move_index = probs.max(1)
        engine_move = get_engine_move_from_index(move_index.data[0])
        move = translate_from_engine_move(engine_move, board.turn)
        move = queen_promotion_if_possible(board, move)
        if self.train:
            return move, m.log_prob(move_index)
        else:
            return move

    def filter_illegal_moves(self, board, probs):
        if self.cuda:
            filtered = Variable(torch.zeros(probs.shape).cuda(
                self.cuda_device))
        else:
            filtered = Variable(torch.zeros(probs.shape))
        for move in board.legal_moves:
            engine_move = translate_to_engine_move(move, board.turn)
            index = get_engine_move_index(engine_move)
            filtered.data[0, index] = probs.data[0, index]
        probs.set_(source=filtered)


def queen_promotion_if_possible(board, move):
    if move.promotion is not None or \
       board.piece_type_at(move.from_square) != chess.PAWN:
        return move

    to_rank = chess.square_rank(move.to_square)
    if to_rank in (0, 7):
        # it's a queen move on a pawn to rank 1 or 8, automatically
        # promote to queen
        move.promotion = chess.QUEEN
        return move
    return move
