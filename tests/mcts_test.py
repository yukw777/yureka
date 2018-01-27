import mcts
import chess
import unittest.mock as mock


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
