import attr
import chess
import math
import time
from board_data import get_board_data, get_reward
from move_translator import (
    translate_to_engine_move,
    get_engine_move_index,
)


@attr.s
class Node():
    children = attr.ib(default=attr.Factory(dict))
    parent = attr.ib(default=None)
    prior = attr.ib(default=0)
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
    terminate_search = attr.ib()
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
        priors = self.policy.get_probs(node.board).squeeze()
        for move in node.board.legal_moves:
            engine_move = translate_to_engine_move(move, node.board.turn)
            index = get_engine_move_index(engine_move)
            prior = priors.data[index]
            node.add_child(move, prior=prior)

    def simulate(self, node):
        if node.children:
            raise MCTSError(node, 'cannot simulate from a non-leaf')
        walker = node
        board = chess.Board(fen=node.board.fen())
        while not board.is_game_over(claim_draw=True):
            move = self.rollout.get_move(board)
            board.push(move)
            walker.add_child(move)
            walker = walker.children[move]

        return walker

    def backup(self, terminal):
        result = terminal.board.result(claim_draw=True)
        reward = get_reward(result, self.root.board.turn)
        value = self.value.get_value(terminal.board)
        walker = terminal
        while walker:
            walker.visit += 1
            walker.result += reward
            walker.value += value
            walker = walker.parent

    def search(self, duration):
        while continue_search(duration):
            leaf = self.select()
            self.expand(leaf)
            terminal = self.simulate(leaf)
            self.backup(terminal)

    def move(self):
        self.search()
        # pick according to the formula


def continue_search(duration):
    # search for {duration} seconds
    remaining = duration
    while remaining >= 0:
        start = time.time()
        yield True
        end = time.time()
        remaining -= end - start
    yield False
