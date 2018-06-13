import attr
import re
import chess
import sys
import torch
import os
import signal

from multiprocessing.managers import BaseManager

from ..learn import models
from ..learn.models.res import ResNet
from ..mcts.networks import (
    PolicyNetwork,
    ValueNetwork,
    ZeroValue,
    RandomPolicy,
)
from ..mcts import Node, MCTS
from ..mcts.constants import DEFAULT_CONFIDENCE
from ..common.utils import print_flush

from . import constants
from .time_manager import TimeManager


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
    model_name = attr.ib(default=constants.DEFAULT_MODEL)
    model_file = attr.ib(default=constants.DEFAULT_MODEL_FILE)
    cuda_device = attr.ib(default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.options = {
            'Model Name': {
                'type': 'string',
                'default': constants.DEFAULT_MODEL,
                'attr_name': 'model_name',
                'py_type': str,
                'model': True,
            },
            'Model File': {
                'type': 'string',
                'default': constants.DEFAULT_MODEL_FILE,
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
        self.engine = PolicyNetwork(
            self.model, train=False, cuda_device=self.cuda_device)
        self.board = chess.Board()

    def new_position(self, fen, moves):
        self.board = chess.Board(fen=fen)
        for uci in moves:
            self.board.push_uci(uci)

    def go(self, args):
        move = self.engine.get_move(self.board)
        print_flush(f'bestmove {move.uci()}')


@attr.s
class UCIMCTSEngine(UCIEngine):
    use_resnet = attr.ib(default=False)
    resnet_name = attr.ib(default=constants.DEFAULT_RESNET)
    resnet_tower_file = attr.ib(default=constants.DEFAULT_RESNET_TOWER_FILE)
    resnet_value_file = attr.ib(default=constants.DEFAULT_RESNET_VALUE_FILE)
    resnet_policy_file = attr.ib(default=constants.DEFAULT_RESNET_POLICY_FILE)
    value_name = attr.ib(default=constants.ZERO_VALUE)
    value_file = attr.ib(default=constants.DEFAULT_VALUE_FILE)
    policy_name = attr.ib(default=constants.RANDOM_POLICY)
    policy_file = attr.ib(default=constants.DEFAULT_POLICY_FILE)
    confidence = attr.ib(default=DEFAULT_CONFIDENCE)
    asynchronous = attr.ib(default=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.options = {
            'Use ResNet': {
                'type': 'check',
                'default': 'true',
                'attr_name': 'use_resnet',
                'py_type': lambda x: x == 'true',
                'model': True,
            },
            'ResNet Name': {
                'type': 'string',
                'default': constants.DEFAULT_RESNET,
                'attr_name': 'resnet_name',
                'py_type': str,
                'model': True,
            },
            'ResNet Tower File': {
                'type': 'string',
                'default': constants.DEFAULT_RESNET_TOWER_FILE,
                'attr_name': 'resnet_tower_file',
                'py_type': str,
                'model': True,
            },
            'ResNet Value File': {
                'type': 'string',
                'default': constants.DEFAULT_RESNET_VALUE_FILE,
                'attr_name': 'resnet_value_file',
                'py_type': str,
                'model': True,
            },
            'ResNet Policy File': {
                'type': 'string',
                'default': constants.DEFAULT_RESNET_POLICY_FILE,
                'attr_name': 'resnet_policy_file',
                'py_type': str,
                'model': True,
            },
            'Value Name': {
                'type': 'string',
                'default': constants.DEFAULT_VALUE,
                'attr_name': 'value_name',
                'py_type': str,
                'model': True,
            },
            'Value File': {
                'type': 'string',
                'default': constants.DEFAULT_VALUE_FILE,
                'attr_name': 'value_file',
                'py_type': str,
                'model': True,
            },
            'Policy Name': {
                'type': 'string',
                'default': constants.DEFAULT_POLICY,
                'attr_name': 'policy_name',
                'py_type': str,
                'model': True,
            },
            'Policy File': {
                'type': 'string',
                'default': constants.DEFAULT_POLICY_FILE,
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
            'Asynchronous': {
                'type': 'check',
                'default': 'true',
                'attr_name': 'asynchronous',
                'py_type': lambda x: x == 'true',
                'model': False,
            },
        }
        signal.signal(signal.SIGALRM, self.stop)
        BaseManager.register('Node', Node)
        self.manager = BaseManager()
        self.manager.start()

    def init_model(self, name, path):
        model = models.create(name)
        model.load_state_dict(torch.load(os.path.expanduser(path)))
        return model

    def init_models(self):
        if self.use_resnet:
            tower, policy, value = models.create(self.resnet_name)
            tower.load_state_dict(
                torch.load(os.path.expanduser(self.resnet_tower_file)))
            policy.load_state_dict(
                torch.load(os.path.expanduser(self.resnet_policy_file)))
            value.load_state_dict(
                torch.load(os.path.expanduser(self.resnet_value_file)))
            self.policy = PolicyNetwork(
                ResNet(tower, policy), train=False)
            self.value = ValueNetwork(ResNet(tower, value))
        else:
            if self.value_name == constants.ZERO_VALUE:
                self.value = ZeroValue()
            else:
                self.value = self.init_model(self.value_name, self.value_file)
                self.value = ValueNetwork(self.value)
            if self.policy_name == constants.RANDOM_POLICY:
                self.policy = RandomPolicy()
            else:
                self.policy = self.init_model(
                    self.policy_name, self.policy_file)
                self.policy = PolicyNetwork(self.policy, train=False)

    def init_engine(self, board=None):
        if board:
            if self.asynchronous:
                root = self.manager.Node(board=board)
            else:
                root = Node(board=board)
        else:
            if self.asynchronous:
                root = self.manager.Node()
            else:
                root = Node()
        self.engine = MCTS(
            root,
            self.value,
            self.policy,
            self.confidence,
            asynchronous=self.asynchronous,
        )
        self.engine.start_search_processes()
        self.time_manager = TimeManager()

    def new_position(self, fen, moves):
        board = chess.Board(fen=fen)
        for uci in moves:
            if board == self.engine.root.get_board():
                self.engine.advance_root(chess.Move.from_uci(uci))
            board.push_uci(uci)
        if board != self.engine.root.get_board():
            self.init_engine(board=board)

    def go(self, args):
        duration = self.time_manager.handle(
            self.engine.root.get_board().turn, args)
        if not duration:
            self.unknown_handler(args)
            return
        print_flush(f'info string search for {duration} seconds')
        self.engine.search()
        signal.setitimer(signal.ITIMER_REAL, duration)

    def stop(self, signum, frame):
        move = self.engine.get_move()
        print_flush(f'bestmove {move.uci()}')
        self.engine.stop_search()
