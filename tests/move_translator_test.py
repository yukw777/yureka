import chess
import move_translator


def test_underpromotion():
    test_cases = [
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.KNIGHT),
            'color': chess.WHITE,
            'expected': 'd7_u_m_n',
        },
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.BISHOP),
            'color': chess.WHITE,
            'expected': 'd7_u_m_b',
        },
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.ROOK),
            'color': chess.WHITE,
            'expected': 'd7_u_m_r',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.KNIGHT),
            'color': chess.WHITE,
            'expected': 'd7_u_lc_n',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.BISHOP),
            'color': chess.WHITE,
            'expected': 'd7_u_lc_b',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.ROOK),
            'color': chess.WHITE,
            'expected': 'd7_u_lc_r',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.KNIGHT),
            'color': chess.WHITE,
            'expected': 'd7_u_rc_n',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.BISHOP),
            'color': chess.WHITE,
            'expected': 'd7_u_rc_b',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.ROOK),
            'color': chess.WHITE,
            'expected': 'd7_u_rc_r',
        },
        {
            'move': chess.Move(chess.D2, chess.D1, promotion=chess.KNIGHT),
            'color': chess.BLACK,
            'expected': 'e7_u_m_n',
        },
        {
            'move': chess.Move(chess.D2, chess.D1, promotion=chess.BISHOP),
            'color': chess.BLACK,
            'expected': 'e7_u_m_b',
        },
        {
            'move': chess.Move(chess.D2, chess.D1, promotion=chess.ROOK),
            'color': chess.BLACK,
            'expected': 'e7_u_m_r',
        },
        {
            'move': chess.Move(chess.D2, chess.C1, promotion=chess.KNIGHT),
            'color': chess.BLACK,
            'expected': 'e7_u_rc_n',
        },
        {
            'move': chess.Move(chess.D2, chess.C1, promotion=chess.BISHOP),
            'color': chess.BLACK,
            'expected': 'e7_u_rc_b',
        },
        {
            'move': chess.Move(chess.D2, chess.C1, promotion=chess.ROOK),
            'color': chess.BLACK,
            'expected': 'e7_u_rc_r',
        },
        {
            'move': chess.Move(chess.D2, chess.E1, promotion=chess.KNIGHT),
            'color': chess.BLACK,
            'expected': 'e7_u_lc_n',
        },
        {
            'move': chess.Move(chess.D2, chess.E1, promotion=chess.BISHOP),
            'color': chess.BLACK,
            'expected': 'e7_u_lc_b',
        },
        {
            'move': chess.Move(chess.D2, chess.E1, promotion=chess.ROOK),
            'color': chess.BLACK,
            'expected': 'e7_u_lc_r',
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
            'expected': 'c7_q_1_n',
        },
        {
            'move': chess.Move(chess.C7, chess.B8, promotion=chess.QUEEN),
            'color': chess.WHITE,
            'expected': 'c7_q_1_nw',
        },
        {
            'move': chess.Move(chess.C7, chess.D8, promotion=chess.QUEEN),
            'color': chess.WHITE,
            'expected': 'c7_q_1_ne',
        },
        {
            'move': chess.Move(chess.C2, chess.C1, promotion=chess.QUEEN),
            'color': chess.BLACK,
            'expected': 'f7_q_1_n',
        },
        {
            'move': chess.Move(chess.C2, chess.B1, promotion=chess.QUEEN),
            'color': chess.BLACK,
            'expected': 'f7_q_1_ne',
        },
        {
            'move': chess.Move(chess.C2, chess.D1, promotion=chess.QUEEN),
            'color': chess.BLACK,
            'expected': 'f7_q_1_nw',
        },
    ]
    for tc in test_cases:
        translated = move_translator.translate_to_engine_move(
            tc['move'], tc['color'])
        assert translated == tc['expected']


def test_queens_move():
    test_cases = [
        {
            'move': chess.Move(chess.D4, chess.D5),
            'color': chess.WHITE,
            'expected': 'd4_q_1_n',
        },
        {
            'move': chess.Move(chess.D4, chess.E5),
            'color': chess.WHITE,
            'expected': 'd4_q_1_ne',
        },
        {
            'move': chess.Move(chess.D4, chess.E4),
            'color': chess.WHITE,
            'expected': 'd4_q_1_e',
        },
        {
            'move': chess.Move(chess.D4, chess.E3),
            'color': chess.WHITE,
            'expected': 'd4_q_1_se',
        },
        {
            'move': chess.Move(chess.D4, chess.D3),
            'color': chess.WHITE,
            'expected': 'd4_q_1_s',
        },
        {
            'move': chess.Move(chess.D4, chess.C3),
            'color': chess.WHITE,
            'expected': 'd4_q_1_sw',
        },
        {
            'move': chess.Move(chess.D4, chess.C4),
            'color': chess.WHITE,
            'expected': 'd4_q_1_w',
        },
        {
            'move': chess.Move(chess.D4, chess.C5),
            'color': chess.WHITE,
            'expected': 'd4_q_1_nw',
        },
        {
            'move': chess.Move(chess.D4, chess.D5),
            'color': chess.BLACK,
            'expected': 'e5_q_1_s',
        },
        {
            'move': chess.Move(chess.D4, chess.E5),
            'color': chess.BLACK,
            'expected': 'e5_q_1_sw',
        },
        {
            'move': chess.Move(chess.D4, chess.E4),
            'color': chess.BLACK,
            'expected': 'e5_q_1_w',
        },
        {
            'move': chess.Move(chess.D4, chess.E3),
            'color': chess.BLACK,
            'expected': 'e5_q_1_nw',
        },
        {
            'move': chess.Move(chess.D4, chess.D3),
            'color': chess.BLACK,
            'expected': 'e5_q_1_n',
        },
        {
            'move': chess.Move(chess.D4, chess.C3),
            'color': chess.BLACK,
            'expected': 'e5_q_1_ne',
        },
        {
            'move': chess.Move(chess.D4, chess.C4),
            'color': chess.BLACK,
            'expected': 'e5_q_1_e',
        },
        {
            'move': chess.Move(chess.D4, chess.C5),
            'color': chess.BLACK,
            'expected': 'e5_q_1_se',
        },
        {
            'move': chess.Move(chess.D4, chess.D7),
            'color': chess.WHITE,
            'expected': 'd4_q_3_n',
        },
        {
            'move': chess.Move(chess.D4, chess.G7),
            'color': chess.WHITE,
            'expected': 'd4_q_3_ne',
        },
        {
            'move': chess.Move(chess.D4, chess.G4),
            'color': chess.WHITE,
            'expected': 'd4_q_3_e',
        },
        {
            'move': chess.Move(chess.D4, chess.G1),
            'color': chess.WHITE,
            'expected': 'd4_q_3_se',
        },
        {
            'move': chess.Move(chess.D4, chess.D1),
            'color': chess.WHITE,
            'expected': 'd4_q_3_s',
        },
        {
            'move': chess.Move(chess.D4, chess.A1),
            'color': chess.WHITE,
            'expected': 'd4_q_3_sw',
        },
        {
            'move': chess.Move(chess.D4, chess.A4),
            'color': chess.WHITE,
            'expected': 'd4_q_3_w',
        },
        {
            'move': chess.Move(chess.D4, chess.A7),
            'color': chess.WHITE,
            'expected': 'd4_q_3_nw',
        },
        {
            'move': chess.Move(chess.D4, chess.D7),
            'color': chess.BLACK,
            'expected': 'e5_q_3_s',
        },
        {
            'move': chess.Move(chess.D4, chess.G7),
            'color': chess.BLACK,
            'expected': 'e5_q_3_sw',
        },
        {
            'move': chess.Move(chess.D4, chess.G4),
            'color': chess.BLACK,
            'expected': 'e5_q_3_w',
        },
        {
            'move': chess.Move(chess.D4, chess.G1),
            'color': chess.BLACK,
            'expected': 'e5_q_3_nw',
        },
        {
            'move': chess.Move(chess.D4, chess.D1),
            'color': chess.BLACK,
            'expected': 'e5_q_3_n',
        },
        {
            'move': chess.Move(chess.D4, chess.A1),
            'color': chess.BLACK,
            'expected': 'e5_q_3_ne',
        },
        {
            'move': chess.Move(chess.D4, chess.A4),
            'color': chess.BLACK,
            'expected': 'e5_q_3_e',
        },
        {
            'move': chess.Move(chess.D4, chess.A7),
            'color': chess.BLACK,
            'expected': 'e5_q_3_se',
        },
    ]
    for tc in test_cases:
        translated = move_translator.translate_to_engine_move(
            tc['move'], tc['color'])
        assert translated == tc['expected']


def test_knights_move():
    test_cases = [
        {
            'move': chess.Move(chess.D4, chess.E6),
            'color': chess.WHITE,
            'expected': 'd4_n_ur',
        },
        {
            'move': chess.Move(chess.D4, chess.F5),
            'color': chess.WHITE,
            'expected': 'd4_n_ru',
        },
        {
            'move': chess.Move(chess.D4, chess.F3),
            'color': chess.WHITE,
            'expected': 'd4_n_rd',
        },
        {
            'move': chess.Move(chess.D4, chess.E2),
            'color': chess.WHITE,
            'expected': 'd4_n_dr',
        },
        {
            'move': chess.Move(chess.D4, chess.C2),
            'color': chess.WHITE,
            'expected': 'd4_n_dl',
        },
        {
            'move': chess.Move(chess.D4, chess.B3),
            'color': chess.WHITE,
            'expected': 'd4_n_ld',
        },
        {
            'move': chess.Move(chess.D4, chess.B5),
            'color': chess.WHITE,
            'expected': 'd4_n_lu',
        },
        {
            'move': chess.Move(chess.D4, chess.C6),
            'color': chess.WHITE,
            'expected': 'd4_n_ul',
        },
        {
            'move': chess.Move(chess.D4, chess.E6),
            'color': chess.BLACK,
            'expected': 'e5_n_dl',
        },
        {
            'move': chess.Move(chess.D4, chess.F5),
            'color': chess.BLACK,
            'expected': 'e5_n_ld',
        },
        {
            'move': chess.Move(chess.D4, chess.F3),
            'color': chess.BLACK,
            'expected': 'e5_n_lu',
        },
        {
            'move': chess.Move(chess.D4, chess.E2),
            'color': chess.BLACK,
            'expected': 'e5_n_ul',
        },
        {
            'move': chess.Move(chess.D4, chess.C2),
            'color': chess.BLACK,
            'expected': 'e5_n_ur',
        },
        {
            'move': chess.Move(chess.D4, chess.B3),
            'color': chess.BLACK,
            'expected': 'e5_n_ru',
        },
        {
            'move': chess.Move(chess.D4, chess.B5),
            'color': chess.BLACK,
            'expected': 'e5_n_rd',
        },
        {
            'move': chess.Move(chess.D4, chess.C6),
            'color': chess.BLACK,
            'expected': 'e5_n_dr',
        },
    ]
    for tc in test_cases:
        translated = move_translator.translate_to_engine_move(
            tc['move'], tc['color'])
        assert translated == tc['expected']
