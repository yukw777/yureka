import pandas as pd
import chess
import chess.pgn
from state_generator import StateGenerator


def test_generate_correct_num_games():
    state_gen = StateGenerator("tests/test.pgn", "bogus")
    assert len(list(state_gen.get_game())) == 2


def test_generate_correct_sq_piece_data():
    state_gen = StateGenerator("tests/test.pgn", "bogus")
    g = next(state_gen.get_game())
    df = pd.DataFrame(state_gen.get_square_piece_data(g))
    assert df.loc[0, 'square_piece'] == (
        'a1-R,b1-N,c1-B,d1-Q,e1-K,f1-B,g1-N,h1-R,a2-P,b2-P,'
        'c2-P,d2-P,e2-P,f2-P,g2-P,h2-P,a7-p,b7-p,c7-p,d7-p,e7-p,'
        'f7-p,g7-p,h7-p,a8-r,b8-n,c8-b,d8-q,e8-k,f8-b,g8-n,h8-r'
    )
    assert df.loc[1, 'square_piece'] == (
        'h8-r,g8-n,f8-b,e8-q,d8-k,c8-b,b8-n,a8-r,h7-p,g7-p,'
        'f7-p,e7-p,c7-p,b7-p,a7-p,d5-p,h2-P,g2-P,f2-P,e2-P,d2-P,'
        'c2-P,b2-P,a2-P,h1-R,g1-N,f1-B,e1-Q,d1-K,c1-B,b1-N,a1-R'
    )


def test_repetition_data():
    b = chess.Board(fen='4k3/8/8/8/8/8/8/4K3 w - - 0 1')

    # Create transpositions
    def move(b):
        b.push(chess.Move.from_uci('e1e2'))
        b.push(chess.Move.from_uci('e8e7'))
        b.push(chess.Move.from_uci('e2e1'))
        b.push(chess.Move.from_uci('e7e8'))
    for i in range(3):
        move(b)

    game = chess.pgn.Game.from_board(b)
    state_gen = StateGenerator("tests/test.pgn", "bogus")  # file not used

    df = pd.DataFrame(state_gen.get_repetition_data(game))

    for i, data in df.iterrows():
        if i < 4:
            # no transpositions yet
            assert data['rep_2'] == 0
            assert data['rep_3'] == 0
        elif i < 8:
            # twofold transpositions
            assert data['rep_2'] == 1
            assert data['rep_3'] == 0
        else:
            # threefold transpositions
            assert data['rep_2'] == 1
            assert data['rep_3'] == 1


def test_turn_data():
    b = chess.Board(fen='4k3/8/8/8/8/8/8/4K3 w - - 0 1')
    b.push(chess.Move.from_uci('e1e2'))
    b.push(chess.Move.from_uci('e8e7'))
    b.push(chess.Move.from_uci('e2e1'))
    b.push(chess.Move.from_uci('e7e8'))

    game = chess.pgn.Game.from_board(b)
    state_gen = StateGenerator("tests/test.pgn", "bogus")  # file not used

    df = pd.DataFrame(state_gen.get_turn_data(game))
    for i, data in df.iterrows():
        if i % 2 == 0:
            assert data['turn'] == 1
        else:
            assert data['turn'] == 0


def test_move_count_data():
    state_gen = StateGenerator("tests/test.pgn", "bogus")  # file not used
    game = next(state_gen.get_game())

    df = pd.DataFrame(state_gen.get_move_count_data(game))
    for i, data in df.iterrows():
        assert int(i / 2) + 1 == data['move_count']


def test_castling_data():
    state_gen = StateGenerator("tests/test.pgn", "bogus")  # file not used

    def get_castling_game(king_side=True):
        b = chess.Board(fen='r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1')
        # hack to reset the castling rights
        b.castling_rights = chess.BB_CORNERS
        if king_side:
            b.push(chess.Move.from_uci('e1g1'))
            b.push(chess.Move.from_uci('e8g8'))
            b.push(chess.Move.from_uci('g1g2'))
        else:
            b.push(chess.Move.from_uci('e1c1'))
            b.push(chess.Move.from_uci('e8c8'))
            b.push(chess.Move.from_uci('c1c2'))
        return chess.pgn.Game.from_board(b)

    test_cases = [
        {
            'name': 'Kingside',
            'game': get_castling_game(),
            'expected_data': pd.DataFrame([
                {
                    'w_kingside_castling': 1,
                    'w_queenside_castling': 1,
                    'b_kingside_castling': 1,
                    'b_queenside_castling': 1,
                },
                {
                    'w_kingside_castling': 0,
                    'w_queenside_castling': 0,
                    'b_kingside_castling': 1,
                    'b_queenside_castling': 1,
                },
                {
                    'w_kingside_castling': 0,
                    'w_queenside_castling': 0,
                    'b_kingside_castling': 0,
                    'b_queenside_castling': 0,
                },
            ]),
        },
        {
            'name': 'Queenside',
            'game': get_castling_game(king_side=False),
            'expected_data': pd.DataFrame([
                {
                    'w_kingside_castling': 1,
                    'w_queenside_castling': 1,
                    'b_kingside_castling': 1,
                    'b_queenside_castling': 1,
                },
                {
                    'w_kingside_castling': 0,
                    'w_queenside_castling': 0,
                    'b_kingside_castling': 1,
                    'b_queenside_castling': 1,
                },
                {
                    'w_kingside_castling': 0,
                    'w_queenside_castling': 0,
                    'b_kingside_castling': 0,
                    'b_queenside_castling': 0,
                },
            ]),
        },
    ]

    for tc in test_cases:
        df = pd.DataFrame(state_gen.get_castling_data(tc['game']))
        assert df.equals(tc['expected_data'])


def test_no_progress_count_data():
    b = chess.Board(fen='4k3/8/8/8/8/8/8/4K3 w - - 0 1')

    # no progress moves
    def move(b):
        b.push(chess.Move.from_uci('e1e2'))
        b.push(chess.Move.from_uci('e8e7'))
        b.push(chess.Move.from_uci('e2e1'))
        b.push(chess.Move.from_uci('e7e8'))
    for i in range(50):
        move(b)

    game = chess.pgn.Game.from_board(b)
    state_gen = StateGenerator("tests/test.pgn", "bogus")  # file not used

    df = pd.DataFrame(state_gen.get_no_progress_data(game))

    for i, data in df.iterrows():
        assert data['no_progress'] == int(i / 2)


def test_move_data():
    b = chess.Board(fen='4k3/8/8/8/8/8/8/4K3 w - - 0 1')
    b.push(chess.Move.from_uci('e1e2'))
    b.push(chess.Move.from_uci('e8e7'))

    game = chess.pgn.Game.from_board(b)
    state_gen = StateGenerator("tests/test.pgn", "bogus")  # file not used

    df = pd.DataFrame(state_gen.get_move_data(game))
    assert df.equals(pd.DataFrame([
        {'move': 'e1_q_1_n'},
        {'move': 'd1_q_1_n'},
    ]))


def test_generate():
    state_gen = StateGenerator("tests/test.pgn", "bogus")
    df = state_gen.generate()

    # square piece data = 1
    # repetition = 2
    # turn (color) = 1
    # move count = 1
    # for each color, king/queen castling = 2 + 2
    # no progress count = 1
    # move = 1
    assert df.shape == (165, 1+2+1+1+2+2+1+1)
