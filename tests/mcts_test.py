import mcts
import chess
import unittest.mock as mock
import pytest
import torch
import time
from torch.autograd import Variable
from move_translator import (
    translate_to_engine_move,
    get_engine_move_index,
)


def test_node_calculations():
    test_cases = [
        {
            'lambda': 0.1,
            'value': 0.7,
            'visit': 4,
            'result': 1,
            'confidence': 4,
            'expected_q': 0.1825,
            'expected_ucb': 8.1825,
        },
        {
            'lambda': 0.9,
            'value': -0.5,
            'visit': 5,
            'result': -1,
            'confidence': 6,
            'expected_q': -0.19,
            'expected_ucb': 9.81,
        },
    ]

    for tc in test_cases:
        n = mcts.Node(
            value=tc['value'],
            visit=tc['visit'],
            result=tc['result'],
        )
        assert n.q(tc['lambda']) == tc['expected_q']
        assert n.ucb(tc['lambda'], tc['confidence'], 100) == tc['expected_ucb']


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
    m = mcts.MCTS(root, '', '', '', '', '')

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
    mock_policy.get_probs.return_value = Variable(probs)
    m = mcts.MCTS('', '', '', mock_policy, '', '')
    # no children at this point
    n = mcts.Node()

    m.expand(n)
    # should have children now. 20 to be exact since we just expanded the root
    # with priors initialized by the policy network
    assert len(n.children) == 20
    for move, c in n.children.items():
        engine_move = translate_to_engine_move(move, chess.WHITE)
        index = get_engine_move_index(engine_move)
        assert c.prior == probs[0, index]

    # can't expand if it already has been expanded
    with pytest.raises(mcts.MCTSError):
        m.expand(n)


def test_simulate():
    # use fool's mate to test
    mock_rollout = mock.MagicMock()
    mock_rollout.get_move.side_effect = [
        chess.Move.from_uci('f2f3'),
        chess.Move.from_uci('e7e5'),
        chess.Move.from_uci('g2g4'),
        chess.Move.from_uci('d8h4'),
    ]
    mock_value = mock.MagicMock()
    mock_value.get_value.return_value = -0.9
    n = mcts.Node()
    m = mcts.MCTS(n, mock_rollout, mock_value, '', '', '')
    reward, value = m.simulate(n)
    assert reward == -1
    assert value == -0.9

    with pytest.raises(mcts.MCTSError):
        n.children[1] = mcts.Node()
        m.simulate(n)


def test_backup():
    node = mcts.Node()
    node.parent = mcts.Node()
    node.parent.parent = mcts.Node()
    m = mcts.MCTS('', '', '', '', '', '')
    m.backup(node, 1, 0.9)
    walker = node
    while walker:
        assert walker.result == 1
        assert walker.value == 0.9
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
