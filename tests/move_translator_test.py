import chess
import move_translator


def test_underpromotion():
    test_cases = [
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.KNIGHT),
            'expected': 'd7_u_m_n',
        },
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.BISHOP),
            'expected': 'd7_u_m_b',
        },
        {
            'move': chess.Move(chess.D7, chess.D8, promotion=chess.ROOK),
            'expected': 'd7_u_m_r',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.KNIGHT),
            'expected': 'd7_u_lc_n',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.BISHOP),
            'expected': 'd7_u_lc_b',
        },
        {
            'move': chess.Move(chess.D7, chess.C8, promotion=chess.ROOK),
            'expected': 'd7_u_lc_r',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.KNIGHT),
            'expected': 'd7_u_rc_n',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.BISHOP),
            'expected': 'd7_u_rc_b',
        },
        {
            'move': chess.Move(chess.D7, chess.E8, promotion=chess.ROOK),
            'expected': 'd7_u_rc_r',
        },
    ]
    for tc in test_cases:
        translated = move_translator.translate_to_engine_move(tc['move'])
        assert translated == tc['expected']
