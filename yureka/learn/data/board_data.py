import chess
import collections

from . import move_translator


pieces = [
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.KNIGHT, chess.WHITE),
    chess.Piece(chess.BISHOP, chess.WHITE),
    chess.Piece(chess.ROOK, chess.WHITE),
    chess.Piece(chess.QUEEN, chess.WHITE),
    chess.Piece(chess.KING, chess.WHITE),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.KNIGHT, chess.BLACK),
    chess.Piece(chess.BISHOP, chess.BLACK),
    chess.Piece(chess.ROOK, chess.BLACK),
    chess.Piece(chess.QUEEN, chess.BLACK),
    chess.Piece(chess.KING, chess.BLACK),
]


BOARD_SIZE = (len(chess.FILE_NAMES), len(chess.RANK_NAMES))


def get_reward(result, color, award_tie=False):
    points = result.split('-')
    if color == chess.WHITE:
        player_point = points[0]
    else:
        player_point = points[1]

    if player_point == '0':
        return -1
    elif player_point == '1/2':
        if award_tie:
            return 0.5
        else:
            return 0
    elif player_point == '1':
        return 1
    else:
        raise Exception(f'Unknown result: {result}, {color}')


def get_board_data(board, color, history=1):
    row = {
        # 1 if white else 0
        'color': 1 if color else 0,
    }
    row.update(get_historical_piece_rep_data(board, color, history))
    row.update(get_move_count_data(board))
    row.update(get_castling_data(board))
    row.update(get_no_progress_data(board))
    return row


def get_no_progress_data(board):
    return {'no_progress': int(board.halfmove_clock / 2)}


def get_castling_data(board):
    w_kingside = 1 if board.has_kingside_castling_rights(
        chess.WHITE) else 0
    w_queenside = 1 if board.has_queenside_castling_rights(
        chess.WHITE) else 0
    b_kingside = 1 if board.has_kingside_castling_rights(
        chess.BLACK) else 0
    b_queenside = 1 if board.has_queenside_castling_rights(
        chess.BLACK) else 0
    return {
        'w_kingside_castling': w_kingside,
        'w_queenside_castling': w_queenside,
        'b_kingside_castling': b_kingside,
        'b_queenside_castling': b_queenside,
    }


def get_move_count_data(board):
    return {'move_count': board.fullmove_number}


def get_repetition_data(board):
    transposition_key = board._transposition_key()
    transpositions = collections.Counter()
    transpositions.update((transposition_key, ))

    # Count positions.
    switchyard = collections.deque()
    while board.move_stack:
        move = board.pop()
        switchyard.append(move)

        if board.is_irreversible(move):
            break

        transpositions.update((board._transposition_key(), ))

    while switchyard:
        board.push(switchyard.pop())

    data_dict = {
        'rep_2': 0,
        'rep_3': 0,
    }
    if transpositions[transposition_key] >= 3:
        # this position repeated at least three times
        data_dict['rep_2'] = 1
        data_dict['rep_3'] = 1
    elif transpositions[transposition_key] >= 2:
        # this position repeated at least twice
        data_dict['rep_2'] = 1
    return data_dict


def get_historical_piece_rep_data(board, color, history):
    data = {}
    copied = board.copy()
    for i in range(history):
        try:
            if i != 0:
                copied.pop()
            piece_data = get_square_piece_data(copied, color)
            repetition_data = get_repetition_data(copied)
        except IndexError:
            # no more history, so everything should be empty
            piece_data = get_square_piece_data(chess.Board.empty(), color)
            repetition_data = get_repetition_data(chess.Board.empty())
        data['white_square_piece_%d' % i] = piece_data['white_square_piece']
        data['black_square_piece_%d' % i] = piece_data['black_square_piece']
        data['rep_2_%d' % i] = repetition_data['rep_2']
        data['rep_3_%d' % i] = repetition_data['rep_3']

    return data


def get_square_piece_data(board, color):
    piece_map = board.piece_map()
    white_data = []
    black_data = []

    for sq, piece in piece_map.items():
        if color == chess.WHITE:
            sq_name_for_player = chess.SQUARE_NAMES[sq]
        else:
            inv_square = move_translator.square_invert(sq)
            sq_name_for_player = chess.SQUARE_NAMES[inv_square]
        key = f'{sq_name_for_player}-{piece.symbol()}'
        if piece.color == chess.WHITE:
            white_data.append(key)
        else:
            black_data.append(key)

    return {
        'white_square_piece': ','.join(white_data),
        'black_square_piece': ','.join(black_data),
    }
