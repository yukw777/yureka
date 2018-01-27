import chess
import collections
import move_translator


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


def get_reward(result, color, award_tie=True):
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


def get_board_data(board):
    row = {}
    row.update(get_square_piece_data(board))
    row.update(get_repetition_data(board))
    row.update(get_turn_data(board))
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


def get_turn_data(board):
    return {'turn': 1 if board.turn else 0}  # 1 if white else 0


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


def get_square_piece_data(board):
    piece_map = board.piece_map()
    white_data = []
    black_data = []
    for sq, sq_name in enumerate(chess.SQUARE_NAMES):
        if board.turn == chess.WHITE:
            sq_name_for_player = sq_name
        else:
            inv_square = move_translator.square_invert(sq)
            sq_name_for_player = chess.SQUARE_NAMES[inv_square]
        for piece in pieces:
            occupied = get_square_piece_value(
                piece_map, sq, piece)

            key = f'{sq_name_for_player}-{piece.symbol()}'
            if occupied:
                if piece.color == chess.WHITE:
                    white_data.append(key)
                else:
                    black_data.append(key)
    return {
        'white_square_piece': ','.join(white_data),
        'black_square_piece': ','.join(black_data),
    }


def get_square_piece_value(piece_map, square, piece):
    p = piece_map.get(square)
    if p and p == piece:
        return True
    else:
        return False
