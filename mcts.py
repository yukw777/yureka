#!/home/keunwoo/Documents/Projects/chess-engine/venv/bin/python

import attr
import chess
import chess_dataset
import multiprocessing as mp
import math
import models
import time
import torch
from torch.autograd import Variable
import random
import os
from board_data import get_board_data, get_reward
from chess_engine import (
    print_flush,
    ChessEngine,
    UCIEngine,
)
from move_translator import (
    TOTAL_MOVES,
    translate_to_engine_move,
    get_engine_move_index,
)


DEFAULT_ROLLOUT = 'Rollout.v0'
DEFAULT_ROLLOUT_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'saved_models',
    'Rollout',
    'Policy_2018-01-31_21:13:39_7.model',
)
RANDOM_POLICY = 'random'
DEFAULT_VALUE = 'Value.v0'
DEFAULT_VALUE_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'saved_models',
    'Value',
    'Value_2018-01-31_14:20:50_4.model',
)
ZERO_VALUE = 'zero'
DEFAULT_POLICY = 'Policy.v0'
DEFAULT_POLICY_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'saved_models',
    'SL_endgame',
    'Policy_2018-01-27_07:09:34_14.model',
)
DEFAULT_LAMBDA = 0
DEFAULT_CONFIDENCE = math.sqrt(2)


@attr.s
class Node():
    children = attr.ib(default=attr.Factory(dict))
    parent = attr.ib(default=None)
    prior = attr.ib(default=0)
    result = attr.ib(default=0)
    value = attr.ib(default=0)
    visit = attr.ib(default=0)
    board = attr.ib(default=chess.Board())

    def q(self, lambda_c):
        if self.visit == 0:
            return math.inf
        q = (1 - lambda_c) * self.value / self.visit
        q += lambda_c * self.result / self.visit
        return q

    def ucb(self, lambda_c, confidence, visit_sum):
        # alpha go version
        ucb = self.q(lambda_c)
        ucb += confidence * self.prior * math.sqrt(visit_sum)
        ucb /= (1 + self.visit)
        return ucb

    def add_child(self, move, **kwargs):
        b = chess.Board(fen=self.board.fen())
        b.push(move)
        self.children[move] = Node(
            parent=self,
            board=b,
            **kwargs
        )


@attr.s
class MCTSError(Exception):
    node = attr.ib()
    message = attr.ib()


@attr.s
class MCTS():
    root = attr.ib()
    rollout = attr.ib()
    value = attr.ib()
    policy = attr.ib()
    lambda_c = attr.ib(default=DEFAULT_LAMBDA)
    confidence = attr.ib(default=DEFAULT_CONFIDENCE)
    board_queue = attr.ib(default=attr.Factory(mp.Queue))
    move_queue = attr.ib(default=attr.Factory(mp.Queue))

    def select(self):
        node = self.root
        while node.children:
            child_nodes = node.children.values()
            visit_sum = sum([n.visit for n in child_nodes])
            node = max(
                child_nodes,
                key=lambda n: n.ucb(self.lambda_c, self.confidence, visit_sum)
            )
        return node

    def expand(self, node):
        if node.children:
            raise MCTSError(node, 'Cannot expand a non-leaf node')
        if node.board.legal_moves:
            priors = self.policy.get_probs(node.board).squeeze()
            for move in node.board.legal_moves:
                engine_move = translate_to_engine_move(move, node.board.turn)
                index = get_engine_move_index(engine_move)
                prior = priors.data[index]
                node.add_child(move, prior=prior)
            return random.choice(list(node.children.values()))
        else:
            # terminal state, just return itself
            return node

    def simulate(self, node):
        if node.children:
            # TODO: needs to be removed if parallel
            raise MCTSError(node, 'cannot simulate from a non-leaf')
        if self.lambda_c == 0:
            # it will be zero either way, so no need to simulate
            reward = 0
        else:
            board = chess.Board(fen=node.board.fen())
            while not board.is_game_over(claim_draw=True):
                move = self.rollout.get_move(board, sample=True)
                board.push(move)

            result = board.result(claim_draw=True)
            reward = get_reward(result, self.root.board.turn)

        return reward

    def calculate_value(self, node):
        if node.children:
            # TODO: needs to be removed if parallel
            raise MCTSError(node, 'cannot simulate from a non-leaf')
        board = chess.Board(fen=node.board.fen())
        if self.lambda_c == 1:
            # it will be zero either way, so no need to calculate
            return 0
        else:
            return self.value.get_value(board)

    def backup(self, node, reward, value):
        walker = node
        while walker:
            walker.visit += 1
            walker.result += reward
            walker.value += value
            walker = walker.parent

    def search(self, duration):
        search_time = continue_search(duration)
        count = 0
        for t in search_time:
            if not t:
                print_flush(f'info string search iterations: {count}')
                break
            leaf = self.select()
            leaf = self.expand(leaf)
            reward = self.simulate(leaf)
            value = self.calculate_value(leaf)
            self.backup(leaf, reward, value)
            count += 1

    def get_move(self):
        # pick the move with the max visit from the root
        if not self.root.children:
            raise MCTSError(self.root, 'You should search before get_move')
        return max(
            self.root.children,
            key=lambda m: self.root.children[m].visit
        )

    def advance_root(self, move):
        child = self.root.children.get(move)
        if child:
            self.root = child
        else:
            self.root.add_child(move)
            self.root = self.root.children[move]
        self.root.parent = None


TC_WTIME = 'wtime'
TC_BTIME = 'btime'
TC_WINC = 'winc'
TC_BINC = 'binc'
TC_MOVESTOGO = 'movestogo'
TC_MOVETIME = 'movetime'
TC_KEYS = [
    TC_WTIME,
    TC_BTIME,
    TC_WINC,
    TC_BINC,
    TC_MOVESTOGO,
    TC_MOVETIME,
]


@attr.s
class TimeManager():
    total_time = attr.ib(default=None)
    total_moves = attr.ib(default=None)

    def handle_movetime(self, data):
        return data[TC_MOVETIME]

    def handle_fischer(self, color, data):
        if color == chess.WHITE:
            time = data[TC_WTIME]
            otime = data[TC_BTIME]
            inc = data[TC_WINC]
        else:
            time = data[TC_BTIME]
            otime = data[TC_WTIME]
            inc = data[TC_BINC]

        ratio = max(otime/time, 1.0)
        # assume we have 16 moves to go
        moves = 16 * min(2.0, ratio)
        return time / moves + 3 / 4 * inc

    def handle_classic(self, color, data):
        if self.total_time is None and self.total_moves is None:
            # first time getting time control information
            # assume this is the start
            self.total_moves = data.get(TC_MOVESTOGO)
            if color == chess.WHITE:
                self.total_time = data[TC_WTIME]
            else:
                self.total_time = data[TC_BTIME]
        if color == chess.WHITE:
            time = data[TC_WTIME]
        else:
            time = data[TC_BTIME]
        moves = data.get(TC_MOVESTOGO, 20)
        tc = time / moves
        if self.total_moves:
            tc_cf = time + self.total_time
            tc_cf /= moves + self.total_moves
        else:
            tc_cf = math.inf
        return min(tc, tc_cf)

    def handle(self, color, args):
        data = parse_time_control(args)
        return self.calculate_duration(color, data)

    def calculate_duration(self, color, data):
        if TC_MOVETIME in data:
            duration = self.handle_movetime(data)
        elif TC_WINC in data and TC_BINC in data:
            duration = self.handle_fischer(color, data)
        else:
            duration = self.handle_classic(color, data)
        return duration / 1000


def continue_search(duration):
    # search for {duration} seconds
    remaining = duration
    while remaining >= 0:
        start = time.time()
        yield True
        end = time.time()
        remaining -= end - start
    yield False


def parse_time_control(args):
    data = {}
    args = args.split()
    for i in range(len(args)):
        token = args[i]
        if token in TC_KEYS:
            data[token] = float(args[i+1])
    return data


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
class ValueNetwork():
    value = attr.ib()
    cuda = attr.ib(default=True)
    cuda_device = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.value.eval()
        self.cuda = self.cuda and torch.cuda.is_available()
        if self.cuda:
            self.value.cuda(self.cuda_device)

    def get_value(self, board):
        board_data = get_board_data(board)
        tensor = chess_dataset.get_tensor_from_row(board_data)
        tensor = tensor.unsqueeze(0)
        value = self.value(Variable(tensor.cuda(), volatile=True))
        return value.squeeze().data[0]


@attr.s
class UCIMCTSEngine(UCIEngine):
    rollout_name = attr.ib(default=DEFAULT_ROLLOUT)
    rollout_file = attr.ib(default=DEFAULT_ROLLOUT_FILE)
    value_name = attr.ib(default=DEFAULT_VALUE)
    value_file = attr.ib(default=DEFAULT_VALUE_FILE)
    policy_name = attr.ib(default=DEFAULT_POLICY)
    policy_file = attr.ib(default=DEFAULT_POLICY_FILE)
    lambda_c = attr.ib(default=DEFAULT_LAMBDA)
    confidence = attr.ib(default=DEFAULT_CONFIDENCE)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.options = {
            'Rollout Name': {
                'type': 'string',
                'default': DEFAULT_ROLLOUT,
                'attr_name': 'rollout_name',
                'py_type': str,
                'model': True,
            },
            'Rollout File': {
                'type': 'string',
                'default': DEFAULT_ROLLOUT_FILE,
                'attr_name': 'rollout_file',
                'py_type': str,
                'model': True,
            },
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
            'Lambda': {
                'type': 'string',
                'default': DEFAULT_LAMBDA,
                'attr_name': 'lambda_c',
                'py_type': float,
                'model': False,
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
        if self.rollout_name == RANDOM_POLICY:
            self.rollout = RandomPolicy()
        else:
            self.rollout = self.init_model(
                self.rollout_name, self.rollout_file)
            self.rollout = ChessEngine(self.rollout, train=False)
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
            self.rollout,
            self.value,
            self.policy,
            self.lambda_c,
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
