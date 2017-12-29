import pytest
import pandas as pd
import chess
from state_generator import StateGenerator, pieces


@pytest.fixture
def state_gen():
    return StateGenerator("tests/test.pgn")


def test_generate_correct_num_games(state_gen):
    assert len(list(state_gen.get_game())) == 2


def check_square(df, w_sq, b_sq, symbols_to_check):
    for p in pieces:
        expected_val = 0
        symbol = p.symbol()
        squares = w_sq if p.color == chess.WHITE else b_sq
        if symbol in symbols_to_check:
            expected_val = 1

        for sq in squares:
            assert df.loc[0, f'{sq}-{symbol}'] == expected_val


def test_generate_correct_sq_piece_data(state_gen):
    g = next(state_gen.get_game())
    df = pd.DataFrame(state_gen.get_square_piece_data(g))
    assert df.shape == (57, 8*8*(6+6))

    # make sure the initial board configuration is correct
    # Kings
    check_square(df, ('e1',), ('e8',), ('K', 'k'))

    # Queens
    check_square(df, ('d1',), ('d8',), ('Q', 'q'))

    # Rooks
    check_square(df, ('a1', 'h1'), ('a8', 'h8'), ('R', 'r'))

    # Bishops
    check_square(df, ('a1', 'h1'), ('a8', 'h8'), ('R', 'r'))

    # Knights
    check_square(df, ('b1', 'g1'), ('b8', 'g8'), ('N', 'n'))

    # Pawns
    check_square(
        df,
        ('a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'),
        ('a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'),
        ('P', 'p')
    )
