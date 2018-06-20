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
            # plain movetime
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
            # regular fischer
            'data': [
                {
                    'wtime': 40000,
                    'btime': 50000,
                    'winc': 1000,
                    'binc': 1000,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 50000,
                    'btime': 60000,
                    'winc': 1000,
                    'binc': 1000,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 50000,
                    'btime': 50000,
                    'winc': 1000,
                    'binc': 1000,
                    'color': chess.BLACK,
                },
            ],
            'expected': [2.75, 4.5, 3.875],
        },
        {
            # panic mode for fischer
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
                    'wtime': 1500,
                    'btime': 20000,
                    'winc': 1000,
                    'binc': 1000,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 1500,
                    'winc': 1000,
                    'binc': 1000,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 1500,
                    'btime': 20000,
                    'winc': 60,
                    'binc': 1000,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 1500,
                    'winc': 1000,
                    'binc': 60,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 700,
                    'btime': 20000,
                    'winc': 50,
                    'binc': 1000,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 700,
                    'winc': 1000,
                    'binc': 50,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 400,
                    'btime': 20000,
                    'winc': 50,
                    'binc': 1000,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 400,
                    'winc': 1000,
                    'binc': 50,
                    'color': chess.BLACK,
                },
            ],
            'expected': [
                0.5,
                1.0,
                0.075,
                0.075,
                0.06,
                0.06,
                0.025,
                0.025,
                0.1,
                0.1,
            ],
        },
        {
            # regular classic
            'data': [
                {
                    'wtime': 100000,
                    'btime': 100000,
                    'movestogo': 25,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 200000,
                    'btime': 200000,
                    'movestogo': 25,
                    'color': chess.BLACK,
                },
            ],
            'expected': [4.0, 6.0],
        },
        {
            # regular classic when we have a lot less time than opponent
            'data': [
                {
                    'wtime': 150000,
                    'btime': 500000,
                    'movestogo': 25,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 1000000,
                    'btime': 250000,
                    'movestogo': 25,
                    'color': chess.BLACK,
                },
            ],
            'expected': [3.0, 4.0],
        },
        {
            # panic mode for fischer
            'data': [
                {
                    'wtime': 10000,
                    'btime': 20000,
                    'movestogo': 10,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 20000,
                    'movestogo': 10,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 1500,
                    'btime': 20000,
                    'movestogo': 10,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 1500,
                    'movestogo': 10,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 1500,
                    'btime': 20000,
                    'movestogo': 10,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 1500,
                    'movestogo': 10,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 700,
                    'btime': 20000,
                    'movestogo': 10,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 700,
                    'movestogo': 10,
                    'color': chess.BLACK,
                },
                {
                    'wtime': 400,
                    'btime': 20000,
                    'movestogo': 10,
                    'color': chess.WHITE,
                },
                {
                    'wtime': 10000,
                    'btime': 400,
                    'movestogo': 10,
                    'color': chess.BLACK,
                },
            ],
            'expected': [
                0.5,
                1.0,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.1,
                0.1,
            ],
        },
    ]

    for tc in test_cases:
        tm = time_manager.TimeManager()
        for d, e in zip(tc['data'], tc['expected']):
            duration = tm.calculate_duration(d['color'], d)
            assert duration == e
