import pandas as pd
import chess
import chess.pgn
import unittest.mock as mock
from state_generator import ExpertStateGenerator, SimSampledStateGenerator


def test_expert_get_correct_num_games():
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")
    assert len(list(state_gen.get_game())) == 2


def test_unbiased_get_game():
    # create mock sl_engine and rl_engine
    step = 6
    sl_engine = mock.MagicMock()
    sl_engine.get_move = mock.Mock(return_value=(1, 2))
    rl_engine = mock.MagicMock()
    rl_engine.get_move = mock.Mock(return_value=(1, 2))
    num_games = 10
    mock_board = mock.MagicMock()
    mock_board.is_game_over.side_effect = ([False] * 9 + [True]) * num_games
    mock_board.turn = chess.WHITE
    mock_board.result.return_value = '1-0'
    with mock.patch('state_generator.chess.Board', return_value=mock_board), \
        mock.patch('state_generator.random.randint', return_value=step), \
            mock.patch('state_generator.random.choice', return_value=1), \
            mock.patch('state_generator.chess.pgn.Game.from_board'):
        state_gen = SimSampledStateGenerator(
            "bogus", sl_engine, rl_engine, num_games)
        games = list(state_gen.get_game())
        assert len(games) == num_games
        for game in games:
            assert len(game) == 3
        assert sl_engine.get_move.call_count == num_games * step
        assert rl_engine.get_move.call_count == num_games * (10 - step - 2)


def test_unbiased_get_game_data():
    with open('tests/test.pgn') as f:
        g = chess.pgn.read_game(f)
    state_gen = SimSampledStateGenerator("bogus", "bogus", "bogus", "bogus")
    step = 10
    data = state_gen.get_game_data((g, step, True))
    assert len(data) == 1


def test_unbiased_get_label_data():
    state_gen = SimSampledStateGenerator("bogus", "bogus", "bogus", "bogus")
    data = state_gen.get_label_data((0, 0, 1))
    assert len(data) == 1
    assert data[0]['value'] == 1


def test_generate_correct_sq_piece_data():
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")
    g = next(state_gen.get_game())
    df = pd.DataFrame(state_gen.get_game_data(g))
    assert df.loc[0, 'white_square_piece'] == (
        'a1-R,b1-N,c1-B,d1-Q,e1-K,f1-B,g1-N,h1-R,a2-P,b2-P,c2-P,'
        'd2-P,e2-P,f2-P,g2-P,h2-P'
    )
    assert df.loc[0, 'black_square_piece'] == (
        'a7-p,b7-p,c7-p,d7-p,e7-p,f7-p,g7-p,h7-p,a8-r,b8-n,c8-b,'
        'd8-q,e8-k,f8-b,g8-n,h8-r'
    )
    assert df.loc[1, 'white_square_piece'] == (
        'h8-R,g8-N,f8-B,e8-Q,d8-K,c8-B,b8-N,a8-R,h7-P,g7-P,f7-P,'
        'e7-P,c7-P,b7-P,a7-P,d5-P'
    )
    assert df.loc[1, 'black_square_piece'] == (
        'h2-p,g2-p,f2-p,e2-p,d2-p,c2-p,b2-p,a2-p,h1-r,g1-n,f1-b,'
        'e1-q,d1-k,c1-b,b1-n,a1-r'
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
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")

    df = pd.DataFrame(state_gen.get_game_data(game))

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
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")

    df = pd.DataFrame(state_gen.get_game_data(game))
    for i, data in df.iterrows():
        if i % 2 == 0:
            assert data['turn'] == 1
        else:
            assert data['turn'] == 0


def test_move_count_data():
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")
    game = next(state_gen.get_game())

    df = pd.DataFrame(state_gen.get_game_data(game))
    for i, data in df.iterrows():
        assert int(i / 2) + 1 == data['move_count']


def test_castling_data():
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")

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
        df = pd.DataFrame(state_gen.get_game_data(tc['game']))
        columns = [
            'b_kingside_castling',
            'b_queenside_castling',
            'w_kingside_castling',
            'w_queenside_castling',
        ]
        assert df[columns].equals(tc['expected_data'])


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
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")

    df = pd.DataFrame(state_gen.get_game_data(game))

    for i, data in df.iterrows():
        assert data['no_progress'] == int(i / 2)


def test_expert_label_data():
    b = chess.Board(fen='4k3/8/8/8/8/8/8/4K3 w - - 0 1')
    b.push(chess.Move.from_uci('e1e2'))
    b.push(chess.Move.from_uci('e8e7'))

    game = chess.pgn.Game.from_board(b)
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")

    df = pd.DataFrame(state_gen.get_label_data(game))
    assert df.equals(pd.DataFrame([
        {'move': 'e1_q_1_n'},
        {'move': 'd1_q_1_n'},
    ]))


def test_generate():
    state_gen = ExpertStateGenerator("bogus", "tests/test.pgn")
    df = state_gen.generate()

    # square piece data = 2
    # repetition = 2
    # turn (color) = 1
    # move count = 1
    # for each color, king/queen castling = 2 + 2
    # no progress count = 1
    # move = 1
    assert df.shape == (165, 2+2+1+1+2+2+1+1)
