import mcts
import chess
import math
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
            'prior': 0.5,
            'expected_q': 0.1825,
            'expected_ucb': 4.0365,
        },
        {
            'lambda': 0.8,
            'value': -0.4,
            'visit': 1,
            'result': -1,
            'confidence': 6,
            'prior': 0.3,
            'expected_q': -0.88,
            'expected_ucb': 8.56,
        },
        {
            'lambda': 0.8,
            'value': -0.4,
            'visit': 0,
            'result': -1,
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
            result=tc['result'],
            prior=tc['prior'],
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
    m = mcts.MCTS(root, '', '', '', '', '', '', parallel=False)

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
    m = mcts.MCTS('', '', '', mock_policy, '', '', parallel=False)
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
    # use fool's mate to test
    mock_rollout = mock.MagicMock()
    mock_rollout.get_move.side_effect = [
        chess.Move.from_uci('f2f3'),
        chess.Move.from_uci('e7e5'),
        chess.Move.from_uci('g2g4'),
        chess.Move.from_uci('d8h4'),
    ]
    n = mcts.Node()
    reward = mcts.simulate(n, 0.5, mock_rollout, n.board.turn)
    assert reward == -1


def test_calculate_value():
    mock_value = mock.MagicMock()
    mock_value.get_value.return_value = -0.9
    n = mcts.Node()
    m = mcts.MCTS(n, '', mock_value, '', '', '', parallel=False)
    value = m.calculate_value(n)
    assert value == -0.9

    with pytest.raises(mcts.MCTSError):
        n.children[1] = mcts.Node()
        m.calculate_value(n)


def test_backup():
    node = mcts.Node()
    node.parent = mcts.Node()
    node.parent.parent = mcts.Node()
    mcts.backup(node, reward=1, value=0.9)
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
    m = mcts.MCTS(root, '', '', '', parallel=False)
    assert m.get_move() == 'm3'

    m = mcts.MCTS(mcts.Node(), '', '', '', parallel=False)
    with pytest.raises(mcts.MCTSError):
        m.get_move()


def test_advance_root():
    children = {i: mcts.Node() for i in range(5)}
    root_with_children = mcts.Node(children=children)
    m = mcts.MCTS(root_with_children, '', '', '', parallel=False)
    m.advance_root(1)
    assert m.root.parent is None
    assert m.root == children[1]

    m = mcts.MCTS(mcts.Node(), '', '', '', parallel=False)
    move = chess.Move.from_uci('f2f3')
    m.advance_root(move)
    assert m.root.parent is None
    assert m.root.board.move_stack == [move]


def test_engine_new_position():
    e = mcts.UCIMCTSEngine(
        rollout_name=mcts.RANDOM_POLICY,
        value_name=mcts.ZERO_VALUE,
        policy_name=mcts.RANDOM_POLICY,
        parallel=False
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


def test_parse_time_control():
    test_cases = [
        {
            'args': 'movetime 10000',
            'data': {
                'movetime': 10000,
            },
        },
        {
            'args': 'wtime 15000 btime 30000 movestogo 15',
            'data': {
                'wtime': 15000,
                'btime': 30000,
                'movestogo': 15,
            },
        },
        {
            'args': 'btime 30000 wtime 15000 movestogo 15',
            'data': {
                'wtime': 15000,
                'btime': 30000,
                'movestogo': 15,
            },
        },
        {
            'args': 'btime 20000 wtime 10000',
            'data': {
                'wtime': 10000,
                'btime': 20000,
            },
        },
        {
            'args': 'wtime 10000 btime 20000',
            'data': {
                'wtime': 10000,
                'btime': 20000,
            },
        },
        {
            'args': 'unknown',
            'data': {},
        },
    ]
    for tc in test_cases:
        parsed = mcts.parse_time_control(tc['args'])
        assert parsed == tc['data']


def test_time_manager():
    test_cases = [
        {
            'data': [
                {
                    'movetime': 10000,
                    'color': chess.WHITE,
                },
                {
                    'movetime': 20000,
                    'color': chess.WHITE,
                },
            ],
            'expected': [10, 20],
        },
        {
            'data': [
                {
                    'wtime': 10000,
                    'btime': 20000,
                    'winc': 1000,
                    'binc': 1000,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 20000,
                    'winc': 1000,
                    'binc': 1000,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 10000,
                    'btime': 30000,
                    'winc': 1000,
                    'binc': 1000,
                    'color': chess.BLACK,
                },
            ],
            'expected': [1.0625, 2.0, 2.625],
        },
        {
            'data': [
                {
                    'wtime': 100000,
                    'btime': 200000,
                    'movestogo': 20,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 100000,
                    'btime': 200000,
                    'movestogo': 20,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 300000,
                    'btime': 200000,
                    'color': chess.WHITE,
                },
            ],
            'expected': [5.0, 7.5, 10],
        },
        {
            'data': [
                {
                    'wtime': 300000,
                    'btime': 200000,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 300000,
                    'btime': 200000,
                    'color': chess.BLACK,
                },
            ],
            'expected': [15.0, 10],
        },
    ]

    for tc in test_cases:
        tm = mcts.TimeManager()
        for d, e in zip(tc['data'], tc['expected']):
            duration = tm.calculate_duration(d['color'], d)
            assert duration == e
