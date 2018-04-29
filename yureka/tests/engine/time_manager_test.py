import chess

from yureka.engine import time_manager


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
        parsed = time_manager.parse_time_control(tc['args'])
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
        tm = time_manager.TimeManager()
        for d, e in zip(tc['data'], tc['expected']):
            duration = tm.calculate_duration(d['color'], d)
            assert duration == e
