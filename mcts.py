import attr
import chess
import chess_engine
import math
import models
import time
import torch
import os
from board_data import get_board_data, get_reward


DEFAULT_ROLLOUT = 'Policy.v0'
DEFAULT_ROLLOUT_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'saved_models',
    'SL_endgame',
    'Policy_2018-01-27_07:09:34_14.model',
)
DEFAULT_VALUE = 'Value.v0'
DEFAULT_VALUE_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'saved_models',
    'Value',
    'Policy_2018-01-27_07:09:34_14.model',
)
ZERO_VALUE = 'zero'
DEFAULT_POLICY = 'Policy.v0'
DEFAULT_POLICY_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'saved_models',
    'SL_endgame',
    'Policy_2018-01-27_07:09:34_14.model',
)
DEFAULT_LAMBDA = 0.5
DEFAULT_CONFIDENCE = 1


@attr.s
class Node():
    children = attr.ib(default=attr.Factory(dict))
    parent = attr.ib(default=None)
    result = attr.ib(default=0)
    value = attr.ib(default=0)
    visit = attr.ib(default=0)
    board = attr.ib(default=chess.Board())

    def __attrs_post_init__(self):
        self.board_data = get_board_data(self.board)

    def q(self, lambda_c):
        q = (1 - lambda_c) * self.value / self.visit
        q += lambda_c * self.result / self.visit
        return q

    def ucb(self, lambda_c, confidence, visit_sum):
        # alpha go version
        ucb = self.q(lambda_c)
        ucb += confidence * math.sqrt(visit_sum) / (1 + self.visit)
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
    lambda_c = attr.ib(default=0.5)
    confidence = attr.ib(default=1)

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
        for move in node.board.legal_moves:
            node.add_child(move)

    def simulate(self, node):
        if node.children:
            raise MCTSError(node, 'cannot simulate from a non-leaf')
        board = chess.Board(fen=node.board.fen())
        while not board.is_game_over(claim_draw=True):
            move = self.rollout.get_move(board, sample=True)
            board.push(move)

        result = board.result(claim_draw=True)
        reward = get_reward(result, self.root.board.turn)
        value = self.value.get_value(board)

        return reward, value

    def backup(self, node, reward, value):
        walker = node
        while walker:
            walker.visit += 1
            walker.result += reward
            walker.value += value
            walker = walker.parent

    def search(self, duration):
        search_time = continue_search(duration)
        for t in search_time:
            if not t:
                break
            leaf = self.select()
            self.expand(leaf)
            reward, value = self.simulate(leaf)
            self.backup(leaf, reward, value)

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


def continue_search(duration):
    # search for {duration} seconds
    remaining = duration
    while remaining >= 0:
        start = time.time()
        yield True
        end = time.time()
        remaining -= end - start
    yield False


class ZeroValue():
    def get_value(self, board):
        return 0


@attr.s
class UCIMCTSEngine(chess_engine.UCIEngine):
    rollout_name = attr.ib(default=DEFAULT_ROLLOUT)
    rollout_file = attr.ib(default=DEFAULT_ROLLOUT_FILE)
    value_name = attr.ib(default=ZERO_VALUE)
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
                'attr_name': 'model_name',
                'py_type': str,
                'model': True,
            },
            'Rollout File': {
                'type': 'string',
                'default': DEFAULT_ROLLOUT_FILE,
                'attr_name': 'model_file',
                'py_type': str,
                'model': True,
            },
            'Value Name': {
                'type': 'string',
                'default': ZERO_VALUE,
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
                'attr_name': 'lambda',
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
        self.rollout = self.init_model(self.rollout_name, self.rollout_file)
        self.rollout = chess_engine.ChessEngine(self.rollout, train=False)
        if self.value_name == ZERO_VALUE:
            self.value = ZeroValue()
        else:
            self.value = self.init_model(self.value_name, self.value_file)
        self.policy = self.init_model(self.policy_name, self.policy_file)
        self.policy = chess_engine.ChessEngine(self.policy, train=False)

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

    def new_position(self, fen, moves):
        board = chess.Board(fen=fen)
        for uci in moves:
            board.push_uci(uci)

        # check if the new board differs by only one move from the current one
        # we have in the engine
        top = board.pop()
        if self.engine.root.board == board:
            # don't throw away the search so far, but advance the root
            self.engine.advance_root(top)
        else:
            board.push(top)
            self.init_engine(board=board)

    def go(self, args):
        self.engine.search(3)
        move = self.engine.get_move()
        print(f'bestmove {move.uci()}')
        self.engine.advance_root(move)


if __name__ == '__main__':
    print('Yureka!')
    UCIMCTSEngine().listen()
