import chess

from yureka.learn.data.board_data import get_board_data
from unittest.mock import patch


def test_get_board_data():
    with patch(
        'yureka.learn.data.board_data.get_square_piece_data',
        return_value={'sq_piece_data': 0}
    ), patch(
        'yureka.learn.data.board_data.get_repetition_data',
        return_value={'rep_data': 0}
    ), patch(
        'yureka.learn.data.board_data.get_move_count_data',
        return_value={'mc_data': 0}
    ), patch(
        'yureka.learn.data.board_data.get_castling_data',
        return_value={'castling': 0}
    ), patch(
        'yureka.learn.data.board_data.get_no_progress_data',
        return_value={'no_progress': 0}
    ):
        expected_row = {
            'sq_piece_data': 0,
            'rep_data': 0,
            'mc_data': 0,
            'castling': 0,
            'no_progress': 0,
        }
        row = get_board_data(chess.Board(), chess.WHITE)
        expected_row['color'] = 1
        assert row == expected_row

        row = get_board_data(chess.Board(), chess.BLACK)
        expected_row['color'] = 0
        assert row == expected_row
