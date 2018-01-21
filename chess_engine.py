#!/home/keunwoo/Documents/Projects/chess-engine/venv/bin/python

import attr
import models
import sys
import os
import collections
import chess
import state_generator
import chess_dataset
import torch
import torch.nn.functional as F
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

        probs = F.softmax(outputs.view(outputs.shape[0], -1), dim=1)
        if self.train:
            # clamp to 1e-12 for numerical stability
            probs = probs.clamp(min=1e-12)
        probs = self.filter_illegal_moves(board, probs)
        if self.train:
            m = Categorical(probs)
            move_index = m.sample()
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
            move_filter = Variable(torch.zeros(probs.shape).cuda(
                self.cuda_device))
        else:
            move_filter = Variable(torch.zeros(probs.shape))
        move_indeces = []
        for move in board.legal_moves:
            engine_move = translate_to_engine_move(move, board.turn)
            index = get_engine_move_index(engine_move)
            move_filter.data[0, index] = 1
            move_indeces.append(index)
        filtered = probs * move_filter
        if not filtered.nonzero().size():
            # all the moves have zero probs. so make it uniform
            # by setting the probs of legal moves to 1
            if self.cuda:
                move_filter = Variable(torch.zeros(probs.shape).cuda(
                    self.cuda_device))
            else:
                move_filter = Variable(torch.zeros(probs.shape))
            for i in move_indeces:
                move_filter.data[0, i] = 1
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


@attr.s
class UCI():
    model = attr.ib()
    model_file = attr.ib()

    def __attrs_post_init__(self):
        self.handlers = {
            'uci': self.uci,
            'isready': self.isready,
            'ucinewgame': self.ucinewgame,
            'position': self.position,
            'go': self.go,
            'quit': self.quit,
        }
        self.model = models.create(self.model)
        self.model.load_state_dict(torch.load(self.model_file))
        self.init_engine()

    def init_engine(self):
        self.engine = ChessEngine(self.model, train=False)
        self.board = chess.Board()

    def uci(self, args):
        print('id name Yureka 0.1')
        print('id author Peter Yu')
        print('uciok')

    def isready(self, args):
        print('readyok')

    def ucinewgame(self, args):
        self.init_engine()

    def position(self, args):
        args = args.split()
        if args[0] == 'startpos':
            fen = chess.STARTING_FEN
            moves = args[2:]
        elif args[0] == 'fen':
            fen = ' '.join(args[1:7])
            moves = args[7:]
        else:
            self.unknown_handler(' '.join(args))
            return
        self.board = chess.Board(fen=fen)
        for uci in moves:
            self.board.push_uci(uci)

    def go(self, args):
        move = self.engine.get_move(self.board)
        print(f'bestmove {move.uci()}')

    def quit(self, args):
        sys.exit()

    def unknown_handler(self, command):
        print(f'Unknown command: {command}')

    def parse_command(self, raw_command):
        parts = raw_command.split(maxsplit=1)
        if len(parts) == 1:
            args = ''
        else:
            args = parts[1]
        return parts[0], args

    def handle(self, raw_command):
        command, args = self.parse_command(raw_command)
        h = self.handlers.get(command)
        if not h:
            self.unknown_handler(raw_command)
        else:
            h(args)

    def listen(self):
        while True:
            command = input()
            self.handle(command)


if __name__ == '__main__':
    import argparse
    default_model = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'saved_models',
        '2018-01-08',
        'ChessEngine_2018-01-09_22:26:13_11.model',
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='ChessEngine.v0')
    parser.add_argument('-f', '--model-file', default=default_model)

    args = parser.parse_args()
    uci = UCI(args.model, args.model_file)
    uci.listen()
