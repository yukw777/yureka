#!/home/keunwoo/Documents/Projects/chess-engine/venv/bin/python

import attr
import chess
import torch
from torch.autograd import Variable
import random
import os
import sys
from .mcts.networks import ValueNetwork
from .mcts import Node, MCTS
from .engine.time_manager import TimeManager
from .mcts.constants import DEFAULT_CONFIDENCE
from . import models
from .chess_engine import (
    print_flush,
    ChessEngine,
    UCIEngine,
)
from .move_translator import (
    TOTAL_MOVES,
    translate_to_engine_move,
    get_engine_move_index,
)


root_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_path)


RANDOM_POLICY = 'random'
DEFAULT_VALUE = 'Value.v0'
DEFAULT_VALUE_FILE = os.path.join(
    root_path,
    'saved_models',
    'Value',
    'Value_2018-01-31_14:20:50_4.model',
)
ZERO_VALUE = 'zero'
DEFAULT_POLICY = 'Policy.v0'
DEFAULT_POLICY_FILE = os.path.join(
    root_path,
    'saved_models',
    'SL_endgame',
    'Policy_2018-01-27_07:09:34_14.model',
)


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


@attr.s
class UCIMCTSEngine(UCIEngine):
    value_name = attr.ib(default=DEFAULT_VALUE)
    value_file = attr.ib(default=DEFAULT_VALUE_FILE)
    policy_name = attr.ib(default=DEFAULT_POLICY)
    policy_file = attr.ib(default=DEFAULT_POLICY_FILE)
    confidence = attr.ib(default=DEFAULT_CONFIDENCE)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.options = {
            'Value Name': {
                'type': 'string',
                'default': DEFAULT_VALUE,
                'attr_name': 'value_name',
                'py_type': str,
                'model': True,
            },
            'Value File': {
                'type': 'string',
                'default': DEFAULT_VALUE_FILE,
                'attr_name': 'value_file',
                'py_type': str,
                'model': True,
            },
            'Policy Name': {
                'type': 'string',
                'default': DEFAULT_POLICY,
                'attr_name': 'policy_name',
                'py_type': str,
                'model': True,
            },
            'Policy File': {
                'type': 'string',
                'default': DEFAULT_POLICY_FILE,
                'attr_name': 'policy_file',
                'py_type': str,
                'model': True,
            },
            'Confidence': {
                'type': 'string',
                'default': DEFAULT_CONFIDENCE,
                'attr_name': 'confidence',
                'py_type': float,
                'model': False,
            },
        }

    def init_model(self, name, path):
        model = models.create(name)
        model.load_state_dict(torch.load(os.path.expanduser(path)))
        return model

    def init_models(self):
        if self.value_name == ZERO_VALUE:
            self.value = ZeroValue()
        else:
            self.value = self.init_model(self.value_name, self.value_file)
            self.value = ValueNetwork(self.value)
        if self.policy_name == RANDOM_POLICY:
            self.policy = RandomPolicy()
        else:
            self.policy = self.init_model(self.policy_name, self.policy_file)
            self.policy = ChessEngine(self.policy, train=False)

    def init_engine(self, board=None):
        if board:
            root = Node(board=board)
        else:
            root = Node()
        self.engine = MCTS(
            root,
            self.value,
            self.policy,
            self.confidence,
        )
        self.time_manager = TimeManager()

    def new_position(self, fen, moves):
        board = chess.Board(fen=fen)
        for uci in moves:
            if board == self.engine.root.board:
                self.engine.advance_root(chess.Move.from_uci(uci))
            board.push_uci(uci)
        if board != self.engine.root.board:
            self.init_engine(board=board)

    def go(self, args):
        duration = self.time_manager.handle(self.engine.root.board.turn, args)
        if not duration:
            self.unknown_handler(args)
            return
        print_flush(f'info string search for {duration} seconds')
        self.engine.search(duration)
        move = self.engine.get_move()
        print_flush(f'bestmove {move.uci()}')


if __name__ == '__main__':
    print_flush('Yureka!')
    UCIMCTSEngine().listen()
