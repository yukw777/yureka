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
QUEEN_MOVE_DIRECTIONS = [
    QUEEN_MOVE_DIRECTION_N,
    QUEEN_MOVE_DIRECTION_NE,
    QUEEN_MOVE_DIRECTION_E,
    QUEEN_MOVE_DIRECTION_SE,
    QUEEN_MOVE_DIRECTION_S,
    QUEEN_MOVE_DIRECTION_SW,
    QUEEN_MOVE_DIRECTION_W,
    QUEEN_MOVE_DIRECTION_NW,
]


KNIGHT_MOVE_PREFIX = 'n'
KNIGHT_MOVE_UP_RIGHT = 'ur'
KNIGHT_MOVE_RIGHT_UP = 'ru'
KNIGHT_MOVE_RIGHT_DOWN = 'rd'
KNIGHT_MOVE_DOWN_RIGHT = 'dr'
KNIGHT_MOVE_DOWN_LEFT = 'dl'
KNIGHT_MOVE_LEFT_DOWN = 'ld'
KNIGHT_MOVE_LEFT_UP = 'lu'
KNIGHT_MOVE_UP_LEFT = 'ul'
KNIGHT_MOVE_DIRECTIONS = [
    KNIGHT_MOVE_UP_RIGHT,
    KNIGHT_MOVE_RIGHT_UP,
    KNIGHT_MOVE_RIGHT_DOWN,
    KNIGHT_MOVE_DOWN_RIGHT,
    KNIGHT_MOVE_DOWN_LEFT,
    KNIGHT_MOVE_LEFT_DOWN,
    KNIGHT_MOVE_LEFT_UP,
    KNIGHT_MOVE_UP_LEFT,
]


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
UNDERPROMOTION_DIRECTIONS = [
    UNDERPROMOTION_PAWN_MOVE,
    UNDERPROMOTION_PAWN_LEFT_CAPTURE,
    UNDERPROMOTION_PAWN_RIGHT_CAPTURE,
]


NUM_MOVE_PLANES = len(QUEEN_MOVE_DIRECTIONS) * 7 + \
    len(KNIGHT_MOVE_DIRECTIONS) + \
    len(UNDERPROMOTION_PIECE_MAP) * len(UNDERPROMOTION_DIRECTIONS)
TOTAL_MOVES = NUM_MOVE_PLANES * len(chess.FILE_NAMES) * len(chess.RANK_NAMES)


BOARD_OFFSET = len(chess.FILE_NAMES) * len(chess.RANK_NAMES)
QUEEN_MOVE_OFFSET = 0
KNIGHT_MOVE_OFFSET = QUEEN_MOVE_OFFSET + len(QUEEN_MOVE_DIRECTIONS) * 7
UNDERPROMOTION_OFFSET = KNIGHT_MOVE_OFFSET + len(KNIGHT_MOVE_DIRECTIONS)


def get_engine_move_from_index(index):
    square, move_offset = get_square_move_offset_from_index(index)
    if move_offset >= UNDERPROMOTION_OFFSET:
        # Underpromotion move
        offset = move_offset - UNDERPROMOTION_OFFSET
        direction, piece = get_underpromotion_from_offset(offset)
        return '_'.join([square, UNDERPROMOTION_PREFIX, direction, piece])
    elif move_offset >= KNIGHT_MOVE_OFFSET:
        # Knight move
        offset = move_offset - KNIGHT_MOVE_OFFSET
        direction = get_knight_from_offset(offset)
        return '_'.join([square, KNIGHT_MOVE_PREFIX, direction])
    else:
        # Queen move
        pass


def get_square_move_offset_from_index(index):
    offset = int(index / BOARD_OFFSET)
    square_index = index % BOARD_OFFSET
    return chess.SQUARE_NAMES[square_index], offset


def get_underpromotion_from_offset(offset):
    direction = offset % len(UNDERPROMOTION_PIECE_MAP)
    direction = UNDERPROMOTION_DIRECTIONS[direction]
    piece = int(offset / len(UNDERPROMOTION_PIECE_MAP))
    piece = chess.PIECE_SYMBOLS[piece + chess.KNIGHT]

    return direction, piece


def get_knight_from_offset(offset):
    return KNIGHT_MOVE_DIRECTIONS[offset]


def square_name_to_square(name):
    f = chess.FILE_NAMES.index(name[0])
    r = chess.RANK_NAMES.index(name[1])
    return chess.square(f, r)


def get_engine_move_index(move):
    sq, move_type, move_data = move.split('_', 2)
    sq_offset = square_name_to_square(sq)
    if move_type == QUEEN_MOVE_PREFIX:
        move_offset = QUEEN_MOVE_OFFSET
        move_offset += get_queen_move_offset(move_data)
    elif move_type == KNIGHT_MOVE_PREFIX:
        move_offset = KNIGHT_MOVE_OFFSET
        move_offset += get_knight_move_offset(move_data)
    elif move_type == UNDERPROMOTION_PREFIX:
        move_offset = UNDERPROMOTION_OFFSET
        move_offset += get_underpromotion_move_offset(move_data)
    else:
        raise Exception(f'Unknown move type: {move_type}')

    return move_offset * BOARD_OFFSET + sq_offset


def get_queen_move_offset(move_data):
    steps, direction = move_data.split('_')
    steps = int(steps) - 1
    direction = QUEEN_MOVE_DIRECTIONS.index(direction)

    # steps X direction
    return steps * len(QUEEN_MOVE_DIRECTIONS) + direction


def get_knight_move_offset(move_data):
    return KNIGHT_MOVE_DIRECTIONS.index(move_data)


def get_underpromotion_move_offset(move_data):
    direction, piece = move_data.split('_')
    direction = UNDERPROMOTION_DIRECTIONS.index(direction)
    piece = chess.PIECE_SYMBOLS.index(piece) - chess.KNIGHT

    # piece X direction
    return piece * len(UNDERPROMOTION_PIECE_MAP) + direction


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
            return get_knight_move(move, color)
        else:
            # queen's move
            return get_queen_move(move, color)


def get_knight_move(move, color):
    if not is_knight_move(move):
        raise Exception('Trying to get a knight move for a non knight move')

    from_rank, from_file, to_rank, to_file = get_ranks_files(move, color)
    if to_rank - from_rank == 2:
        if from_file > to_file:
            knight_move = KNIGHT_MOVE_UP_LEFT
        else:
            knight_move = KNIGHT_MOVE_UP_RIGHT
    elif to_rank - from_rank == 1:
        if from_file > to_file:
            knight_move = KNIGHT_MOVE_LEFT_UP
        else:
            knight_move = KNIGHT_MOVE_RIGHT_UP
    elif to_rank - from_rank == -1:
        if from_file > to_file:
            knight_move = KNIGHT_MOVE_LEFT_DOWN
        else:
            knight_move = KNIGHT_MOVE_RIGHT_DOWN
    else:
        if from_file > to_file:
            knight_move = KNIGHT_MOVE_DOWN_LEFT
        else:
            knight_move = KNIGHT_MOVE_DOWN_RIGHT

    return '_'.join([
        chess.SQUARE_NAMES[get_from_square(move, color)],
        KNIGHT_MOVE_PREFIX,
        knight_move,
    ])


def get_underpromotion_move(move, color):
    direction = get_queen_move_direction(move, color)
    return '_'.join([
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
        chess.SQUARE_NAMES[get_from_square(move, color)],
        QUEEN_MOVE_PREFIX,
        str(chess.square_distance(move.from_square, move.to_square)),
        get_queen_move_direction(move, color),
    ])


def get_ranks_files(move, color):
    from_square = move.from_square
    to_square = move.to_square
    if color == chess.BLACK:
        from_square = square_invert(from_square)
        to_square = square_invert(to_square)

    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)

    return from_rank, from_file, to_rank, to_file


def get_queen_move_direction(move, color):
    if is_knight_move(move):
        raise Exception(
            'Cannot figure out queen move direction of a knight move')

    from_rank, from_file, to_rank, to_file = get_ranks_files(move, color)

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
