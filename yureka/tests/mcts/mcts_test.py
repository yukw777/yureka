import chess
import math
import unittest.mock as mock
import pytest
import torch
import time
from yureka import mcts
from yureka.engine import UCIMCTSEngine
from yureka.engine.constants import ZERO_VALUE, RANDOM_POLICY
from yureka.learn.data.move_translator import (
    translate_to_engine_move,
    get_engine_move_index,
)


def test_node_calculations():
    test_cases = [
        {
            'value': 0.7,
            'visit': 4,
            'confidence': 4,
            'prior': 0.5,
            'expected_q': 0.175,
            'expected_ucb': 4.035,
        },
        {
            'value': -0.4,
            'visit': 1,
            'confidence': 6,
            'prior': 0.3,
            'expected_q': -0.4,
            'expected_ucb': 8.8,
        },
        {
            'value': -0.4,
            'visit': 0,
            'confidence': 6,
            'prior': 0.3,
            'expected_q': math.inf,
            'expected_ucb': math.inf,
        },
    ]

    for tc in test_cases:
        n = mcts.Node(
            value=tc['value'],
            visit=tc['visit'],
            prior=tc['prior'],
        )
        assert n.q() == tc['expected_q']
        assert n.ucb(tc['confidence'], 100) == tc['expected_ucb']


def test_node_add_child():
    root = mcts.Node()
    assert root.children == {}
    root.add_child(chess.Move.from_uci('a2a4'), prior=0.5)
    root.add_child(chess.Move.from_uci('b2b4'), prior=0.3)

    child1 = root.children[chess.Move.from_uci('a2a4')]
    child2 = root.children[chess.Move.from_uci('b2b4')]
    assert child1.prior == 0.5
    assert child1.parent == root
    assert child2.prior == 0.3
    assert child2.parent == root

    b = chess.Board()
    b.push_uci('a2a4')
    assert child1.board == b
    b = chess.Board()
    b.push_uci('b2b4')
    assert child2.board == b
    b = chess.Board()
    assert root.board == b


def test_select():
    children = []
    for i in range(3):
        n = mcts.Node()
        n.ucb = mock.MagicMock(return_value=i)
        children.append(n)
    root = mcts.Node()
    m = mcts.MCTS(root, '', '', '')

    # if root is already a leaf, return that
    selected = m.select()
    assert root == selected

    # traverse the tree, picking the node with the biggest ucb
    root.children = {i: children[i] for i in range(3)}
    selected = m.select()
    assert selected == children[-1]


def test_expand():
    mock_policy = mock.MagicMock()
    probs = torch.randn(1, 4672)
    mock_policy.get_probs.return_value = probs
    m = mcts.MCTS('', '', mock_policy, '')
    # no children at this point
    n = mcts.Node()

    random_child = m.expand(n)
    # should have children now. 20 to be exact since we just expanded the root
    assert len(n.children) == 20
    for move, c in n.children.items():
        engine_move = translate_to_engine_move(move, chess.WHITE)
        index = get_engine_move_index(engine_move)
        assert c.prior == probs[0, index]
    assert random_child in n.children.values()

    # can't expand if it already has been expanded
    with pytest.raises(mcts.MCTSError):
        m.expand(n)

    # if expanding a terminal state, just return the node
    n = mcts.Node(board=chess.Board(
        fen='3b1q1q/1N2PRQ1/rR3KBr/B4PP1/2Pk1r1b/1P2P1N1/2P2P2/8 '
            'b - - 0 1'))
    assert n == m.expand(n)


def test_simulate():
    mock_value = mock.MagicMock()
    mock_value.get_value.return_value = -0.9
    n = mcts.Node()
    m = mcts.MCTS(n, mock_value, '', '')
    value = m.simulate(n)
    assert value == -0.9

    with pytest.raises(mcts.MCTSError):
        n.children[1] = mcts.Node()
        m.simulate(n)


def test_simulate_game_over():
    # if the game is over, return the reward
    # use the fool's mate
    n = mcts.Node()
    n.board = chess.Board(
        fen='rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3')
    m = mcts.MCTS(n, '', '', '')
    value = m.simulate(n)
    assert value == -1


def test_backup():
    # white turn
    node = mcts.Node()
    # black turn
    node.parent = mcts.Node(board=chess.Board(
        fen='rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1'))
    # white turn
    node.parent.parent = mcts.Node()
    m = mcts.MCTS('', '', '', '')
    m.backup(node, 0.9)
    walker = node
    while walker:
        if walker.board.turn == node.board.turn:
            assert walker.value == 0.9
        else:
            assert walker.value == -0.9
        assert walker.visit == 1
        walker = walker.parent


def test_continue_search():
    count = 0
    search_time = mcts.continue_search(1.5)
    for t in search_time:
        if not t:
            break
        count += 1
        time.sleep(1)
    assert count == 2


def test_get_move():
    children = {
        'm1': mcts.Node(visit=1),
        'm2': mcts.Node(visit=0),
        'm3': mcts.Node(visit=3),
        'm4': mcts.Node(visit=2),
    }
    root = mcts.Node(children=children)
    m = mcts.MCTS(root, '', '', '')
    assert m.get_move() == 'm3'

    m = mcts.MCTS(mcts.Node(), '', '', '')
    with pytest.raises(mcts.MCTSError):
        m.get_move()


def test_advance_root():
    children = {i: mcts.Node() for i in range(5)}
    root_with_children = mcts.Node(children=children)
    m = mcts.MCTS(root_with_children, '', '', '')
    m.advance_root(1)
    assert m.root.parent is None
    assert m.root == children[1]

    m = mcts.MCTS(mcts.Node(), '', '', '')
    move = chess.Move.from_uci('f2f3')
    m.advance_root(move)
    assert m.root.parent is None
    assert m.root.board.move_stack == [move]


def test_engine_new_position():
    e = UCIMCTSEngine(
        value_name=ZERO_VALUE,
        policy_name=RANDOM_POLICY,
    )
    e.init_models()
    e.init_engine()
    e.engine.expand(e.engine.root)
    e.engine.expand(e.engine.root.children[chess.Move.from_uci('e2e4')])

    e.new_position(chess.STARTING_FEN, ['e2e4'])
    expected = chess.Board()
    expected.push_uci('e2e4')
    assert e.engine.root.board == expected
    assert e.engine.root.parent is None
    assert len(e.engine.root.children) != 0

    e.new_position(chess.STARTING_FEN, ['a2a4'])
    expected = chess.Board()
    expected.push_uci('a2a4')
    assert e.engine.root.board == expected
    assert e.engine.root.parent is None
    assert len(e.engine.root.children) == 0
