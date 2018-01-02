import pandas as pd
import chess
import chess.pgn
from state_generator import StateGenerator, pieces


def test_generate_correct_num_games():
    state_gen = StateGenerator("tests/test.pgn", "bogus")
    assert len(list(state_gen.get_game())) == 2


def check_square(df, player_sq, opponent_sq, symbols_to_check, step, turn):
    for p in pieces:
        expected_val = 0
        symbol = p.symbol()
        if turn == chess.BLACK:
            symbol = symbol.swapcase()
        squares = player_sq if p.color == turn else opponent_sq
        if symbol in symbols_to_check:
            expected_val = 1

        for sq in squares:
            assert df.loc[step, f'{sq}-{symbol}'] == expected_val


def test_generate_correct_sq_piece_data():
    state_gen = StateGenerator("tests/test.pgn", "bogus")
    g = next(state_gen.get_game())
    df = pd.DataFrame(state_gen.get_square_piece_data(g))
    assert df.shape == (57, 8*8*(6+6))

    # make sure the initial board configuration is correct
    # Kings
    check_square(df, ('e1',), ('e8',), ('K', 'k'), 0, chess.WHITE)

    # Queens
    check_square(df, ('d1',), ('d8',), ('Q', 'q'), 0, chess.WHITE)

    # Rooks
    check_square(df, ('a1', 'h1'), ('a8', 'h8'), ('R', 'r'), 0, chess.WHITE)

    # Bishops
    check_square(df, ('c1', 'f1'), ('c8', 'f8'), ('B', 'b'), 0, chess.WHITE)

    # Knights
    check_square(df, ('b1', 'g1'), ('b8', 'g8'), ('N', 'n'), 0, chess.WHITE)

    # Pawns
    check_square(
        df,
        ('a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'),
        ('a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'),
        ('P', 'p'),
        0,
        chess.WHITE
    )

    # Board configuration after one move e2e4 (white pawn to e4)
    # Kings
    check_square(df, ('d1',), ('d8',), ('k', 'K'), 1, chess.BLACK)

    # Queens
    check_square(df, ('e1',), ('e8',), ('q', 'Q'), 1, chess.BLACK)

    # Rooks
    check_square(df, ('a1', 'h1'), ('a8', 'h8'), ('r', 'R'), 1, chess.BLACK)

    # Bishops
    check_square(df, ('c1', 'f1'), ('c8', 'f8'), ('b', 'B'), 1, chess.BLACK)

    # Knights
    check_square(df, ('b1', 'g1'), ('b8', 'g8'), ('n', 'N'), 1, chess.BLACK)

    # Pawns
    check_square(
        df,
        ('a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'),
        ('a7', 'b7', 'c7', 'd5', 'e7', 'f7', 'g7', 'h7'),
        ('p', 'P'),
        1,
        chess.BLACK
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

    # for each square, each piece and each color = 8*8*6*2
    # repetition = 2
    # turn (color) = 1
    # move count = 1
    # for each color, king/queen castling = 2 + 2
    # no progress count = 1
    # move = 1
    assert df.shape == (165, 8*8*6*2+2+1+1+2+2+1+1)
