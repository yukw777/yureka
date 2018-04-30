import attr
import chess
import math
import time
import random

from ..board_data import get_reward
from ..utils import print_flush
from ..move_translator import (
    translate_to_engine_move,
    get_engine_move_index,
)

from .errors import MCTSError
from .constants import DEFAULT_CONFIDENCE


@attr.s
class Node():
    children = attr.ib(default=attr.Factory(dict))
    parent = attr.ib(default=None)
    prior = attr.ib(default=0)
    value = attr.ib(default=0)
    visit = attr.ib(default=0)
    board = attr.ib(default=chess.Board())

    def q(self):
        if self.visit == 0:
            return math.inf
        return self.value / self.visit

    def ucb(self, confidence, visit_sum):
        # alpha go version
        ucb = self.q()
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
class MCTS():
    root = attr.ib()
    value = attr.ib()
    policy = attr.ib()
    confidence = attr.ib(default=DEFAULT_CONFIDENCE)

    def select(self):
        node = self.root
        while node.children:
            child_nodes = node.children.values()
            visit_sum = sum([n.visit for n in child_nodes])
            node = max(
                child_nodes,
                key=lambda n: n.ucb(self.confidence, visit_sum)
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
            raise MCTSError(node, 'cannot simulate from a non-leaf')
        board = chess.Board(fen=node.board.fen())
        if board.is_game_over(claim_draw=True):
            return get_reward(board.result(claim_draw=True), board.turn)
        return self.value.get_value(board)

    def backup(self, node, value):
        walker = node
        turn = node.board.turn
        while walker:
            walker.visit += 1
            if walker.board.turn == turn:
                walker.value += value
            else:
                walker.value -= value
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
            value = self.simulate(leaf)
            self.backup(leaf, value)
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


def continue_search(duration):
    # search for {duration} seconds
    remaining = duration
    while remaining >= 0:
        start = time.time()
        yield True
        end = time.time()
        remaining -= end - start
    yield False
