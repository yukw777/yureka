import chess


QUEEN_MOVE_PREFIX = 'q'
QUEEN_MOVE_DIRECTION_N = 'n'
QUEEN_MOVE_DIRECTION_NE = 'ne'
QUEEN_MOVE_DIRECTION_E = 'e'
QUEEN_MOVE_DIRECTION_SE = 'se'
QUEEN_MOVE_DIRECTION_S = 's'
QUEEN_MOVE_DIRECTION_SW = 'sw'
QUEEN_MOVE_DIRECTION_W = 'w'
QUEEN_MOVE_DIRECTION_NW = 'nw'


KNIGHT_MOVE_PREFIX = 'n'
KNIGHT_MOVE_UP_RIGHT = 'ur'
KNIGHT_MOVE_RIGHT_UP = 'ru'
KNIGHT_MOVE_RIGHT_DOWN = 'rd'
KNIGHT_MOVE_DOWN_RIGHT = 'dr'
KNIGHT_MOVE_DOWN_LEFT = 'dl'
KNIGHT_MOVE_LEFT_DOWN = 'ld'
KNIGHT_MOVE_LEFT_UP = 'lu'
KNIGHT_MOVE_UP_LEFT = 'ul'


UNDERPROMOTION_PREFIX = 'u'
UNDERPROMOTION_PAWN_MOVE = 'm'
UNDERPROMOTION_PAWN_LEFT_CAPTURE = 'lc'
UNDERPROMOTION_PAWN_RIGHT_CAPTURE = 'rc'
UNDERPROMOTION_KNIGHT = chess.Piece(chess.KNIGHT, chess.BLACK).symbol()
UNDERPROMOTION_BISHOP = chess.Piece(chess.BISHOP, chess.BLACK).symbol()
UNDERPROMOTION_ROOK = chess.Piece(chess.ROOK, chess.BLACK).symbol()
UNDERPROMOTION_PIECE_MAP = {
    chess.KNIGHT: UNDERPROMOTION_KNIGHT,
    chess.BISHOP: UNDERPROMOTION_BISHOP,
    chess.ROOK: UNDERPROMOTION_ROOK,
}


def translate_to_engine_move(move):
    if move.promotion:
        if move.promotion != chess.QUEEN:
            # underpromotion
            from_square_file = chess.square_file(move.from_square)
            to_square_file = chess.square_file(move.to_square)
            if from_square_file == to_square_file:
                pawn_move = UNDERPROMOTION_PAWN_MOVE
            elif from_square_file < to_square_file:
                pawn_move = UNDERPROMOTION_PAWN_RIGHT_CAPTURE
            else:
                pawn_move = UNDERPROMOTION_PAWN_LEFT_CAPTURE
            return '_'.join([
                chess.SQUARE_NAMES[move.from_square],
                UNDERPROMOTION_PREFIX,
                pawn_move,
                UNDERPROMOTION_PIECE_MAP[move.promotion],
            ])
        else:
            # queen's move
            return
    else:
        if True:
            # queen's move
            return
        else:
            # knight's move
            return
