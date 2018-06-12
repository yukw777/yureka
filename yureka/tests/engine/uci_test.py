import chess

from yureka.engine import UCIPolicyEngine
from yureka.engine.constants import DEFAULT_MODEL, DEFAULT_MODEL_FILE


def test_uci_setoption():
    test_cases = [
        {
            'args': 'name Model File value file_name',
            'attr': 'model_file',
            'value': 'file_name',
        },
        {
            'args': 'name Model Name value model_name',
            'attr': 'model_name',
            'value': 'model_name',
        },
        {
            'args': 'name CUDA Device value 2',
            'attr': 'cuda_device',
            'value': 2,
        },
        {
            'args': 'name Unknown value 2',
            'attr': 'cuda_device',
            'value': None,
        },
        {
            'args': 'name Unknown value 2',
            'attr': 'model_file',
            'value': DEFAULT_MODEL_FILE,
        },
        {
            'args': 'name Unknown value 2',
            'attr': 'model_name',
            'value': DEFAULT_MODEL,
        },
    ]
    for tc in test_cases:
        uci = UCIPolicyEngine()
        uci.setoption(tc['args'])
        assert getattr(uci, tc['attr']) == tc['value']


def test_uci_position():
    test_cases = [
        {
            'args': 'startpos moves e2e4 c7c5 b1c3 b8c6 g1f3 g7g6',
            'board': chess.Board(),
            'moves': ['e2e4', 'c7c5', 'b1c3', 'b8c6', 'g1f3', 'g7g6'],
        },
        {
            'args': 'startpos',
            'board': chess.Board(),
            'moves': [],
        },
        {
            'args': 'fen r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R'
                    ' b KQkq - 3 3',
            'board': chess.Board(fen='r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/'
                                 'PPPP1PPP/R1BQKB1R b KQkq - 3 3'),
            'moves': [],
        },
        {
            'args': 'fen r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R'
                    ' b KQkq - 3 3 moves a7a5 a2a4',
            'board': chess.Board(fen='r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N2N2/'
                                 'PPPP1PPP/R1BQKB1R b KQkq - 3 3'),
            'moves': ['a7a5', 'a2a4'],
        },
    ]

    for tc in test_cases:
        uci = UCIPolicyEngine()
        uci.position(tc['args'])
        for m in tc['moves']:
            tc['board'].push_uci(m)
        assert uci.board == tc['board']
