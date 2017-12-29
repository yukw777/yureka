import pytest
import pandas as pd
import chess
from state_generator import StateGenerator, pieces


@pytest.fixture
def state_gen():
    return StateGenerator("tests/test.pgn")


def test_generate_correct_num_games(state_gen):
    assert len(list(state_gen.get_game())) == 2


def test_generate_correct_sq_piece_data(state_gen):
    g = next(state_gen.get_game())
    df = pd.DataFrame(state_gen.get_square_piece_data(g))
    assert df.shape == (57, 8*8*(6+6))

    # make sure the initial board configuration is correct
    # Kings
    for p in pieces:
        expected_val = 0
        symbol = p.symbol()
        square = 'e1' if p.color == chess.WHITE else 'e8'
        if symbol in ('K', 'k'):
            expected_val = 1

        assert df.loc[0, f'{square}-{symbol}'] == expected_val

    # Queens
    assert df.loc[0, 'd1-Q'] == 1
    assert df.loc[0, 'd8-q'] == 1

    # Rooks

    # Bishops

    # Knights

    # Pawns
