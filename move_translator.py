import chess


MOVE_COLOR_WHITE = 'w'
MOVE_COLOR_BLACK = 'b'
MOVE_COLOR_MAP = {
    chess.WHITE: MOVE_COLOR_WHITE,
    chess.BLACK: MOVE_COLOR_BLACK,
}


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
UNDERPROMOTION_DIRECTION_MAP = {
    QUEEN_MOVE_DIRECTION_N: UNDERPROMOTION_PAWN_MOVE,
    QUEEN_MOVE_DIRECTION_NE: UNDERPROMOTION_PAWN_RIGHT_CAPTURE,
    QUEEN_MOVE_DIRECTION_NW: UNDERPROMOTION_PAWN_LEFT_CAPTURE,
}


def translate_to_engine_move(move, color):
    if move.promotion:
        if move.promotion != chess.QUEEN:
            # underpromotion
            return get_underpromotion_move(move, color)
        else:
            # queen's move
            return get_queen_move(move, color)
    else:
        if is_knight_move(move):
            # knight's move
            return
        else:
            # queen's move
            return


def get_underpromotion_move(move, color):
    direction = get_queen_move_direction(move, color)
    return '_'.join([
        MOVE_COLOR_MAP[color],
        chess.SQUARE_NAMES[get_from_square(move, color)],
        UNDERPROMOTION_PREFIX,
        UNDERPROMOTION_DIRECTION_MAP[direction],
        UNDERPROMOTION_PIECE_MAP[move.promotion],
    ])


def get_from_square(move, color):
    from_square = move.from_square
    if color == chess.BLACK:
        # always from the perspective of the current player
        from_square = square_invert(from_square)

    return from_square


def get_queen_move(move, color):
    return '_'.join([
        MOVE_COLOR_MAP[color],
        chess.SQUARE_NAMES[get_from_square(move, color)],
        QUEEN_MOVE_PREFIX,
        str(chess.square_distance(move.from_square, move.to_square)),
        get_queen_move_direction(move, color),
    ])


def get_queen_move_direction(move, color):
    if is_knight_move(move):
        raise Exception(
            'Cannot figure out queen move direction of a knight move')

    from_square = move.from_square
    to_square = move.to_square
    if color == chess.BLACK:
        from_square = square_invert(from_square)
        to_square = square_invert(to_square)

    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)

    if from_rank == to_rank:
        if from_file > to_file:
            # west
            return QUEEN_MOVE_DIRECTION_W
        else:
            # east
            return QUEEN_MOVE_DIRECTION_E
    elif from_rank > to_rank:
        # south
        if from_file > to_file:
            # west
            return QUEEN_MOVE_DIRECTION_SW
        elif from_file == to_file:
            return QUEEN_MOVE_DIRECTION_S
        else:
            # east
            return QUEEN_MOVE_DIRECTION_SE
    else:
        # north
        if from_file > to_file:
            # west
            return QUEEN_MOVE_DIRECTION_NW
        elif from_file == to_file:
            return QUEEN_MOVE_DIRECTION_N
        else:
            # east
            return QUEEN_MOVE_DIRECTION_NE


def is_knight_move(move):
    if chess.square_distance(move.from_square, move.to_square) == 2:
        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        # it's a knight move if the diff of rank or file is 1
        if abs(to_rank - from_rank) == 1 or abs(to_file - from_file) == 1:
            return True

    return False


def square_invert(square):
    return square ^ 0x3f
