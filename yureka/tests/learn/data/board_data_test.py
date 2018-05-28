import chess

from yureka.learn.data import board_data
from unittest.mock import patch


def test_get_board_data():
    with patch(
        'yureka.learn.data.board_data.get_historical_piece_rep_data',
        return_value={'historical_sq_rep_data': 0}
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
            'historical_sq_rep_data': 0,
            'mc_data': 0,
            'castling': 0,
            'no_progress': 0,
        }
        row = board_data.get_board_data(chess.Board(), chess.WHITE)
        expected_row['color'] = 1
        assert row == expected_row

        row = board_data.get_board_data(chess.Board(), chess.BLACK)
        expected_row['color'] = 0
        assert row == expected_row


def test_get_historical_piece_rep_data():
    board_with_moves = chess.Board()
    board_with_moves.push_san('e4')
    board_with_moves.push_san('e5')
    board_with_moves.push_san('Nf3')
    board_with_moves.push_san('Nc6')
    board_with_moves.push_san('Bb5')
    board_with_moves.push_san('a6')
    test_cases = [
        {
            'board': chess.Board(),
            'history': 8,
            'color': chess.WHITE,
            'expected_data': {
                'white_square_piece_0':
                    'h2-P,g2-P,f2-P,e2-P,d2-P,c2-P,b2-P,'
                    'a2-P,h1-R,g1-N,f1-B,e1-K,d1-Q,c1-B,b1-N,a1-R',
                'black_square_piece_0':
                    'h8-r,g8-n,f8-b,e8-k,d8-q,c8-b,b8-n,'
                    'a8-r,h7-p,g7-p,f7-p,e7-p,d7-p,c7-p,b7-p,a7-p',
                'rep_2_0': 0,
                'rep_3_0': 0,
                'white_square_piece_1': '',
                'black_square_piece_1': '',
                'rep_2_1': 0,
                'rep_3_1': 0,
                'white_square_piece_2': '',
                'black_square_piece_2': '',
                'rep_2_2': 0,
                'rep_3_2': 0,
                'white_square_piece_3': '',
                'black_square_piece_3': '',
                'rep_2_3': 0,
                'rep_3_3': 0,
                'white_square_piece_4': '',
                'black_square_piece_4': '',
                'rep_2_4': 0,
                'rep_3_4': 0,
                'white_square_piece_5': '',
                'black_square_piece_5': '',
                'rep_2_5': 0,
                'rep_3_5': 0,
                'white_square_piece_6': '',
                'black_square_piece_6': '',
                'rep_2_6': 0,
                'rep_3_6': 0,
                'white_square_piece_7': '',
                'black_square_piece_7': '',
                'rep_2_7': 0,
                'rep_3_7': 0,
            },
        },
        {
            'board': chess.Board(),
            'history': 8,
            'color': chess.BLACK,
            'expected_data': {
                'white_square_piece_0':
                    'a7-P,b7-P,c7-P,d7-P,e7-P,f7-P,g7-P,'
                    'h7-P,a8-R,b8-N,c8-B,d8-K,e8-Q,f8-B,g8-N,h8-R',
                'black_square_piece_0':
                    'a1-r,b1-n,c1-b,d1-k,e1-q,f1-b,g1-n,'
                    'h1-r,a2-p,b2-p,c2-p,d2-p,e2-p,f2-p,g2-p,h2-p',
                'rep_2_0': 0,
                'rep_3_0': 0,
                'white_square_piece_1': '',
                'black_square_piece_1': '',
                'rep_2_1': 0,
                'rep_3_1': 0,
                'white_square_piece_2': '',
                'black_square_piece_2': '',
                'rep_2_2': 0,
                'rep_3_2': 0,
                'white_square_piece_3': '',
                'black_square_piece_3': '',
                'rep_2_3': 0,
                'rep_3_3': 0,
                'white_square_piece_4': '',
                'black_square_piece_4': '',
                'rep_2_4': 0,
                'rep_3_4': 0,
                'white_square_piece_5': '',
                'black_square_piece_5': '',
                'rep_2_5': 0,
                'rep_3_5': 0,
                'white_square_piece_6': '',
                'black_square_piece_6': '',
                'rep_2_6': 0,
                'rep_3_6': 0,
                'white_square_piece_7': '',
                'black_square_piece_7': '',
                'rep_2_7': 0,
                'rep_3_7': 0,
            },
        },
        {
            'board': board_with_moves,
            'history': 8,
            'color': chess.WHITE,
            'expected_data': {
                'white_square_piece_0':
                    'b5-B,e4-P,f3-N,h2-P,g2-P,f2-P,d2-P,c2-P,b2-P,'
                    'a2-P,h1-R,e1-K,d1-Q,c1-B,b1-N,a1-R',
                'black_square_piece_0':
                    'h8-r,g8-n,f8-b,e8-k,d8-q,c8-b,a8-r,h7-p,g7-p,'
                    'f7-p,d7-p,c7-p,b7-p,c6-n,a6-p,e5-p',
                'rep_2_0': 0,
                'rep_3_0': 0,
                'white_square_piece_1':
                    'b5-B,e4-P,f3-N,h2-P,g2-P,f2-P,d2-P,c2-P,b2-P,'
                    'a2-P,h1-R,e1-K,d1-Q,c1-B,b1-N,a1-R',
                'black_square_piece_1':
                    'h8-r,g8-n,f8-b,e8-k,d8-q,c8-b,a8-r,h7-p,g7-p,'
                    'f7-p,d7-p,c7-p,b7-p,a7-p,c6-n,e5-p',
                'rep_2_1': 0,
                'rep_3_1': 0,
                'white_square_piece_2':
                    'e4-P,f3-N,h2-P,g2-P,f2-P,d2-P,c2-P,b2-P,a2-P,'
                    'h1-R,f1-B,e1-K,d1-Q,c1-B,b1-N,a1-R',
                'black_square_piece_2':
                    'h8-r,g8-n,f8-b,e8-k,d8-q,c8-b,a8-r,h7-p,g7-p,'
                    'f7-p,d7-p,c7-p,b7-p,a7-p,c6-n,e5-p',
                'rep_2_2': 0,
                'rep_3_2': 0,
                'white_square_piece_3':
                    'e4-P,f3-N,h2-P,g2-P,f2-P,d2-P,c2-P,b2-P,a2-P,'
                    'h1-R,f1-B,e1-K,d1-Q,c1-B,b1-N,a1-R',
                'black_square_piece_3':
                    'h8-r,g8-n,f8-b,e8-k,d8-q,c8-b,b8-n,a8-r,h7-p,'
                    'g7-p,f7-p,d7-p,c7-p,b7-p,a7-p,e5-p',
                'rep_2_3': 0,
                'rep_3_3': 0,
                'white_square_piece_4':
                    'e4-P,h2-P,g2-P,f2-P,d2-P,c2-P,b2-P,a2-P,h1-R,'
                    'g1-N,f1-B,e1-K,d1-Q,c1-B,b1-N,a1-R',
                'black_square_piece_4':
                    'h8-r,g8-n,f8-b,e8-k,d8-q,c8-b,b8-n,a8-r,h7-p,'
                    'g7-p,f7-p,d7-p,c7-p,b7-p,a7-p,e5-p',
                'rep_2_4': 0,
                'rep_3_4': 0,
                'white_square_piece_5':
                    'e4-P,h2-P,g2-P,f2-P,d2-P,c2-P,b2-P,a2-P,h1-R,'
                    'g1-N,f1-B,e1-K,d1-Q,c1-B,b1-N,a1-R',
                'black_square_piece_5':
                    'h8-r,g8-n,f8-b,e8-k,d8-q,c8-b,b8-n,a8-r,h7-p,'
                    'g7-p,f7-p,e7-p,d7-p,c7-p,b7-p,a7-p',
                'rep_2_5': 0,
                'rep_3_5': 0,
                'white_square_piece_6':
                    'h2-P,g2-P,f2-P,e2-P,d2-P,c2-P,b2-P,a2-P,h1-R,'
                    'g1-N,f1-B,e1-K,d1-Q,c1-B,b1-N,a1-R',
                'black_square_piece_6':
                    'h8-r,g8-n,f8-b,e8-k,d8-q,c8-b,b8-n,a8-r,h7-p,'
                    'g7-p,f7-p,e7-p,d7-p,c7-p,b7-p,a7-p',
                'rep_2_6': 0,
                'rep_3_6': 0,
                'white_square_piece_7': '',
                'black_square_piece_7': '',
                'rep_2_7': 0,
                'rep_3_7': 0,
            },
        },
        {
            'board': board_with_moves,
            'history': 8,
            'color': chess.BLACK,
            'expected_data': {
                'white_square_piece_0':
                    'g4-B,d5-P,c6-N,a7-P,b7-P,c7-P,e7-P,f7-P,g7-P,'
                    'h7-P,a8-R,d8-K,e8-Q,f8-B,g8-N,h8-R',
                'black_square_piece_0':
                    'a1-r,b1-n,c1-b,d1-k,e1-q,f1-b,h1-r,a2-p,b2-p,'
                    'c2-p,e2-p,f2-p,g2-p,f3-n,h3-p,d4-p',
                'rep_2_0': 0,
                'rep_3_0': 0,
                'white_square_piece_1':
                    'g4-B,d5-P,c6-N,a7-P,b7-P,c7-P,e7-P,f7-P,g7-P,'
                    'h7-P,a8-R,d8-K,e8-Q,f8-B,g8-N,h8-R',
                'black_square_piece_1':
                    'a1-r,b1-n,c1-b,d1-k,e1-q,f1-b,h1-r,a2-p,b2-p,'
                    'c2-p,e2-p,f2-p,g2-p,h2-p,f3-n,d4-p',
                'rep_2_1': 0,
                'rep_3_1': 0,
                'white_square_piece_2':
                    'd5-P,c6-N,a7-P,b7-P,c7-P,e7-P,f7-P,g7-P,h7-P,'
                    'a8-R,c8-B,d8-K,e8-Q,f8-B,g8-N,h8-R',
                'black_square_piece_2':
                    'a1-r,b1-n,c1-b,d1-k,e1-q,f1-b,h1-r,a2-p,b2-p,'
                    'c2-p,e2-p,f2-p,g2-p,h2-p,f3-n,d4-p',
                'rep_2_2': 0,
                'rep_3_2': 0,
                'white_square_piece_3':
                    'd5-P,c6-N,a7-P,b7-P,c7-P,e7-P,f7-P,g7-P,h7-P,'
                    'a8-R,c8-B,d8-K,e8-Q,f8-B,g8-N,h8-R',
                'black_square_piece_3':
                    'a1-r,b1-n,c1-b,d1-k,e1-q,f1-b,g1-n,h1-r,a2-p,'
                    'b2-p,c2-p,e2-p,f2-p,g2-p,h2-p,d4-p',
                'rep_2_3': 0,
                'rep_3_3': 0,
                'white_square_piece_4':
                    'd5-P,a7-P,b7-P,c7-P,e7-P,f7-P,g7-P,h7-P,a8-R,'
                    'b8-N,c8-B,d8-K,e8-Q,f8-B,g8-N,h8-R',
                'black_square_piece_4':
                    'a1-r,b1-n,c1-b,d1-k,e1-q,f1-b,g1-n,h1-r,a2-p,'
                    'b2-p,c2-p,e2-p,f2-p,g2-p,h2-p,d4-p',
                'rep_2_4': 0,
                'rep_3_4': 0,
                'white_square_piece_5':
                    'd5-P,a7-P,b7-P,c7-P,e7-P,f7-P,g7-P,h7-P,a8-R,'
                    'b8-N,c8-B,d8-K,e8-Q,f8-B,g8-N,h8-R',
                'black_square_piece_5':
                    'a1-r,b1-n,c1-b,d1-k,e1-q,f1-b,g1-n,h1-r,a2-p,'
                    'b2-p,c2-p,d2-p,e2-p,f2-p,g2-p,h2-p',
                'rep_2_5': 0,
                'rep_3_5': 0,
                'white_square_piece_6':
                    'a7-P,b7-P,c7-P,d7-P,e7-P,f7-P,g7-P,h7-P,a8-R,'
                    'b8-N,c8-B,d8-K,e8-Q,f8-B,g8-N,h8-R',
                'black_square_piece_6':
                    'a1-r,b1-n,c1-b,d1-k,e1-q,f1-b,g1-n,h1-r,a2-p,'
                    'b2-p,c2-p,d2-p,e2-p,f2-p,g2-p,h2-p',
                'rep_2_6': 0,
                'rep_3_6': 0,
                'white_square_piece_7': '',
                'black_square_piece_7': '',
                'rep_2_7': 0,
                'rep_3_7': 0,
            },
        },
    ]

    for tc in test_cases:
        data = board_data.get_historical_piece_rep_data(
            tc['board'], tc['color'], history=tc['history'])
        assert data == tc['expected_data']
