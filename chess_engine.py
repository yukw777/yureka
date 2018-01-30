#!/home/keunwoo/Documents/Projects/chess-engine/venv/bin/python

import attr
import models
import sys
import os
import re
import chess
import chess_dataset
import numpy as np
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
from board_data import get_board_data


DEFAULT_MODEL = 'Policy.v0'
DEFAULT_MODEL_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'saved_models',
    'SL_endgame',
    'Policy_2018-01-27_07:09:34_14.model',
)


@attr.s
class ChessEngine():
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
class UCIEngine():
    def __attrs_post_init__(self):
        self.handlers = {
            'uci': self.uci,
            'isready': self.isready,
            'ucinewgame': self.ucinewgame,
            'position': self.position,
            'go': self.go,
            'stop': self.stop,
            'setoption': self.setoption,
            'quit': self.quit,
        }
        self.model_option_changed = True
        self.engine_option_changed = True

    def init_engine(self):
        raise NotImplemented

    def init_models(self):
        raise NotImplemented

    def uci(self, args):
        print_flush('id name Yureka 0.1')
        print_flush('id author Peter Yu')
        self.print_options()
        print_flush('uciok')

    def print_options(self):
        for name, option in self.options.items():
            print_flush(f"option name {name} type {option['type']} default"
                        f" {option['default']}")

    def setoption(self, args):
        m = re.match(r'name\s+(.+)\s+value\s+(.+)', args)
        if m:
            name = m.group(1)
            value = m.group(2)
        else:
            self.unknown_handler(args)
            return
        option = self.options.get(name)
        if not option:
            return
        setattr(self, option['attr_name'], option['py_type'](value))
        if option['model']:
            self.model_option_changed = True
        else:
            self.engine_option_changed = True

    def stop(self, args):
        pass

    def isready(self, args):
        if self.model_option_changed:
            self.init_models()
            self.model_option_changed = False
        if self.engine_option_changed:
            self.init_engine()
            self.engine_option_changed = False
        print_flush('readyok')

    def ucinewgame(self, args):
        self.init_engine()

    def position(self, args):
        m = re.match(r'startpos(\s+moves\s+(.+))?', args)
        if m:
            fen = chess.STARTING_FEN
            if m.group(2):
                moves = m.group(2).split()
            else:
                moves = []
        else:
            m = re.match(r'fen\s+(.+)\s+moves\s+(.+)', args)
            if m:
                fen = m.group(1)
                moves = m.group(2).split()
            else:
                m = re.match(r'fen\s+(.+)', args)
                if m:
                    fen = m.group(1)
                    moves = []
                else:
                    self.unknown_handler(args)
                    return

        self.new_position(fen, moves)

    def new_position(self, fen, moves):
        raise NotImplemented

    def go(self, args):
        raise NotImplemented

    def quit(self, args):
        sys.exit()

    def unknown_handler(self, command):
        print_flush(f'Unknown command: {command}')

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


@attr.s
class UCIPolicyEngine(UCIEngine):
    model_name = attr.ib(default=DEFAULT_MODEL)
    model_file = attr.ib(default=DEFAULT_MODEL_FILE)
    cuda_device = attr.ib(default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.options = {
            'Model Name': {
                'type': 'string',
                'default': DEFAULT_MODEL,
                'attr_name': 'model_name',
                'py_type': str,
                'model': True,
            },
            'Model File': {
                'type': 'string',
                'default': DEFAULT_MODEL_FILE,
                'attr_name': 'model_file',
                'py_type': str,
                'model': True,
            },
            'CUDA Device': {
                'type': 'string',
                'default': '0',
                'attr_name': 'cuda_device',
                'py_type': int,
                'model': True,
            },
        }
        self.model = None

    def init_models(self):
        self.model = models.create(self.model_name)
        self.model.load_state_dict(
            torch.load(os.path.expanduser(self.model_file)))

    def init_engine(self):
        self.engine = ChessEngine(
            self.model, train=False, cuda_device=self.cuda_device)
        self.board = chess.Board()

    def new_position(self, fen, moves):
        self.board = chess.Board(fen=fen)
        for uci in moves:
            self.board.push_uci(uci)

    def go(self, args):
        move = self.engine.get_move(self.board)
        print_flush(f'bestmove {move.uci()}')


def print_flush(*args, **kwargs):
    print(*args, flush=True, **kwargs)


if __name__ == '__main__':
    import argparse
    default_model = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'saved_models',
        'RL',
        'ChessEngine_2018-01-23_19:22:59_1000.model',
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL)
    parser.add_argument('-f', '--model-file', default=DEFAULT_MODEL_FILE)
    parser.add_argument('-c', '--cuda-device', type=int)

    args = parser.parse_args()
    print('Yureka!')
    uci = UCIPolicyEngine(
        model_name=args.model,
        model_file=os.path.expanduser(args.model_file),
        cuda_device=args.cuda_device
    )
    uci.listen()
