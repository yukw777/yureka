import pytest
from state_generator import StateGenerator


@pytest.fixture
def state_gen():
    return StateGenerator("tests/test.pgn")


def test_generate_correct_num_games(state_gen):
    assert len(list(state_gen.get_game())) == 2
