import chess
import move_translator


def test_underpromotion():
    test_cases = [
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.KNIGHT),
            'color': chess.WHITE,
            'expected': 'w_d7_u_m_n',
        },
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.BISHOP),
            'color': chess.WHITE,
            'expected': 'w_d7_u_m_b',
        },
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.ROOK),
            'color': chess.WHITE,
            'expected': 'w_d7_u_m_r',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.KNIGHT),
            'color': chess.WHITE,
            'expected': 'w_d7_u_lc_n',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.BISHOP),
            'color': chess.WHITE,
            'expected': 'w_d7_u_lc_b',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.ROOK),
            'color': chess.WHITE,
            'expected': 'w_d7_u_lc_r',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.KNIGHT),
            'color': chess.WHITE,
            'expected': 'w_d7_u_rc_n',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.BISHOP),
            'color': chess.WHITE,
            'expected': 'w_d7_u_rc_b',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.ROOK),
            'color': chess.WHITE,
            'expected': 'w_d7_u_rc_r',
        },
        {
            'move': chess.Move(chess.D2, chess.D1, promotion=chess.KNIGHT),
            'color': chess.BLACK,
            'expected': 'b_e7_u_m_n',
        },
        {
            'move': chess.Move(chess.D2, chess.D1, promotion=chess.BISHOP),
            'color': chess.BLACK,
            'expected': 'b_e7_u_m_b',
        },
        {
            'move': chess.Move(chess.D2, chess.D1, promotion=chess.ROOK),
            'color': chess.BLACK,
            'expected': 'b_e7_u_m_r',
        },
        {
            'move': chess.Move(chess.D2, chess.C1, promotion=chess.KNIGHT),
            'color': chess.BLACK,
            'expected': 'b_e7_u_rc_n',
        },
        {
            'move': chess.Move(chess.D2, chess.C1, promotion=chess.BISHOP),
            'color': chess.BLACK,
            'expected': 'b_e7_u_rc_b',
        },
        {
            'move': chess.Move(chess.D2, chess.C1, promotion=chess.ROOK),
            'color': chess.BLACK,
            'expected': 'b_e7_u_rc_r',
        },
        {
            'move': chess.Move(chess.D2, chess.E1, promotion=chess.KNIGHT),
            'color': chess.BLACK,
            'expected': 'b_e7_u_lc_n',
        },
        {
            'move': chess.Move(chess.D2, chess.E1, promotion=chess.BISHOP),
            'color': chess.BLACK,
            'expected': 'b_e7_u_lc_b',
        },
        {
            'move': chess.Move(chess.D2, chess.E1, promotion=chess.ROOK),
            'color': chess.BLACK,
            'expected': 'b_e7_u_lc_r',
        },
    ]
    for tc in test_cases:
        translated = move_translator.translate_to_engine_move(
            tc['move'], tc['color'])
        assert translated == tc['expected']


def test_promotion_queens_move():
    test_cases = [
        {
            'move': chess.Move(chess.C7, chess.C8, promotion=chess.QUEEN),
            'color': chess.WHITE,
            'expected': 'w_c7_q_1_n',
        },
        {
            'move': chess.Move(chess.C7, chess.B8, promotion=chess.QUEEN),
            'color': chess.WHITE,
            'expected': 'w_c7_q_1_nw',
        },
        {
            'move': chess.Move(chess.C7, chess.D8, promotion=chess.QUEEN),
            'color': chess.WHITE,
            'expected': 'w_c7_q_1_ne',
        },
        {
            'move': chess.Move(chess.C2, chess.C1, promotion=chess.QUEEN),
            'color': chess.BLACK,
            'expected': 'b_f7_q_1_n',
        },
        {
            'move': chess.Move(chess.C2, chess.B1, promotion=chess.QUEEN),
            'color': chess.BLACK,
            'expected': 'b_f7_q_1_ne',
        },
        {
            'move': chess.Move(chess.C2, chess.D1, promotion=chess.QUEEN),
            'color': chess.BLACK,
            'expected': 'b_f7_q_1_nw',
        },
    ]
    for tc in test_cases:
        translated = move_translator.translate_to_engine_move(
            tc['move'], tc['color'])
        assert translated == tc['expected']
