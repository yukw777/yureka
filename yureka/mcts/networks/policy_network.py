import attr
import sys
import chess
import numpy as np
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Categorical

from ...chess_dataset import get_tensor_from_row
from ...move_translator import (
    translate_to_engine_move,
    translate_from_engine_move,
    get_engine_move_from_index,
    get_engine_move_index,
)
from ...board_data import get_board_data


@attr.s
class PolicyNetwork():
    model = attr.ib()
    cuda = attr.ib(default=True)
    cuda_device = attr.ib(default=None)
    train = attr.ib(default=True)

    def __attrs_post_init__(self):
        self.cuda = self.cuda and torch.cuda.is_available()
        if self.cuda:
            self.model.cuda(self.cuda_device)
        if self.train:
            self.model.train()
        else:
            self.model.eval()

    def get_probs(self, board):
        board_data = get_board_data(board)
        tensor = get_tensor_from_row(board_data)
        tensor = tensor.view(1, *tensor.shape)
        volatile = not self.model.training
        if self.cuda:
            inputs = Variable(tensor.cuda(self.cuda_device), volatile=volatile)
        else:
            inputs = Variable(tensor, volatile=volatile)
        outputs = self.model(inputs)

        probs = F.softmax(outputs.view(outputs.shape[0], -1), dim=1)
        if self.train:
            # clamp to 1e-12 for numerical stability
            probs = probs.clamp(min=1e-12)
        return self.filter_illegal_moves(board, probs)

    def get_move(self, board, sample=False):
        probs = self.get_probs(board)
        if self.train or sample:
            m = Categorical(probs)
            while True:
                move_index = m.sample()
                if probs.squeeze()[move_index].data[0] == 0:
                    print('Categorical sampled a move with zero prob...',
                          file=sys.stderr)
                    print(f'move_index: {move_index}', file=sys.stderr)
                    nonzero_indeces = probs.squeeze().nonzero().squeeze()
                    print(f'nonzero prob indeces: {nonzero_indeces}')
                    nonzero = probs.squeeze().gather(0, nonzero_indeces)
                    print(f'nonzero probs: {nonzero}', file=sys.stderr)
                else:
                    break
        else:
            _, move_index = probs.max(1)
        engine_move = get_engine_move_from_index(move_index.data[0])
        move = translate_from_engine_move(engine_move, board.turn)
        move = queen_promotion_if_possible(board, move)
        if self.train:
            log_prob = m.log_prob(move_index)
            log_prob_num = log_prob.data[0]
            if np.isnan(log_prob_num) or np.isinf(log_prob_num):
                print('log prob is not a right value!', file=sys.stderr)
                print(f'log_prob: {log_prob}', file=sys.stderr)
                print(f'move_index: {move_index}', file=sys.stderr)
                nonzero_indeces = probs.squeeze().nonzero().squeeze()
                print(f'nonzero prob indeces: {nonzero_indeces}')
                nonzero = probs.squeeze().gather(0, nonzero_indeces)
                print(f'nonzero probs: {nonzero}', file=sys.stderr)
            return move, log_prob
        else:
            return move

    def filter_illegal_moves(self, board, probs):
        if self.cuda:
            move_filter = Variable(torch.zeros(probs.shape).cuda(
                self.cuda_device))
        else:
            move_filter = Variable(torch.zeros(probs.shape))
        move_indeces = []
        for move in board.legal_moves:
            engine_move = translate_to_engine_move(move, board.turn)
            index = get_engine_move_index(engine_move)
            move_indeces.append(index)
        if self.cuda:
            indeces = Variable(torch.LongTensor(move_indeces)).cuda(
                self.cuda_device)
        else:
            indeces = Variable(torch.LongTensor(move_indeces))
        move_filter.index_fill_(1, indeces, 1)

        filtered = probs * move_filter
        if not filtered.nonzero().size():
            # all the moves have zero probs. so make it uniform
            # by setting the probs of legal moves to 1
            if self.cuda:
                move_filter = Variable(torch.zeros(probs.shape).cuda(
                    self.cuda_device))
            else:
                move_filter = Variable(torch.zeros(probs.shape))
            move_filter.index_fill_(1, indeces, 1)
            filtered = filtered + move_filter
        return filtered


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
