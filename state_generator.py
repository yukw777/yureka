import attr
import random
import collections
import chess
import chess.pgn
import pandas as pd
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


def get_reward(result, color):
    points = result.split('-')
    if color == chess.WHITE:
        player_point = points[0]
    else:
        player_point = points[1]

    if player_point == '0':
        return -1
    elif player_point == '1/2':
        return 0
    elif player_point == '1':
        return 1
    else:
        raise Exception(f'Unknown result: {result}, {color}')


def get_board_data(board, transpositions):
    row = {}
    row.update(get_square_piece_data(board))
    row.update(get_repetition_data(board, transpositions))
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


def get_repetition_data(board, transpositions):
    key = board._transposition_key()
    transpositions.update((key, ))
    data_dict = {
        'rep_2': 0,
        'rep_3': 0,
    }
    if transpositions[key] >= 3:
        # this position repeated at least three times
        data_dict['rep_2'] = 1
        data_dict['rep_3'] = 1
    elif transpositions[key] >= 2:
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


@attr.s
class StateGenerator():
    out_csv_file = attr.ib()

    def get_game(self):
        raise NotImplemented

    def get_label_data(self):
        raise NotImplemented

    def get_game_data(self, game):
        raise NotImplemented

    def generate(self, write=False):
        count = 0
        df = pd.DataFrame()
        header = True
        for game in self.get_game():
            count += 1
            game_df = pd.DataFrame(self.get_game_data(game))
            game_df = pd.concat([
                game_df,
                pd.DataFrame(self.get_label_data(game))
            ], axis=1)
            df = pd.concat([df, game_df])
            if count % 100 == 0:
                if write:
                    df.to_csv(
                        self.out_csv_file,
                        index=False,
                        header=header,
                        mode='a'
                    )
                    header = False
                    df = pd.DataFrame()
                print(f'{count} games processed...')
        if write:
            df.to_csv(
                self.out_csv_file,
                index=False,
                header=header,
                mode='a'
            )

        return df


@attr.s
class UnbiasedStateGenerator(StateGenerator):
    sl_engine = attr.ib()
    rl_engine = attr.ib()
    num_games = attr.ib()

    def get_game(self):
        for i in range(self.num_games):
            while True:
                # based on this statistics
                # https://chess.stackexchange.com/questions/2506/what-is-the-average-length-of-a-game-of-chess
                step = random.randint(1, 100)
                board = chess.Board()
                t = 1
                while not board.is_game_over(claim_draw=True):
                    if t < step:
                        move, _ = self.sl_engine.get_move()
                    elif t == step:
                        move = random.choice(list(board.legal_moves))
                        color = board.turn
                    else:
                        move, _ = self.rl_engine.get_move()
                    board.push(move)
                    t += 1
                if t <= step:
                    print(f'We drew {step} steps but the game only got to {t}')
                    print("Let's try again")
                else:
                    break
            result = board.result(claim_draw=True)
            reward = get_reward(result, color)
            yield chess.pgn.Game.from_board(board), step, reward

    def get_game_data(self, data):
        game, step, _ = data
        board = game.board()
        transpositions = collections.Counter()
        t = 1
        for move in game.main_line():
            if t == step + 1:
                return [get_board_data(board, transpositions)]
            board.push(move)
            t += 1

    def get_label_data(self, data):
        _, _, reward = data
        return [{'value': reward}]


@attr.s
class ExpertStateGenerator(StateGenerator):
    game_file_name = attr.ib()

    def __attrs_post_init__(self):
        self.game_file = open(self.game_file_name, 'r')

    def get_game(self):
        while True:
            g = chess.pgn.read_game(self.game_file)
            if g is None:
                break
            yield g

    def get_game_data(self, game):
        board = game.board()
        transpositions = collections.Counter()
        for move in game.main_line():
            yield get_board_data(board, transpositions)
            board.push(move)

    def get_label_data(self, game):
        board = game.board()
        for move in game.main_line():
            yield {'move': move_translator.translate_to_engine_move(
                move, board.turn)}
            board.push(move)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pgn_file')
    parser.add_argument('out_csv_file')
    args = parser.parse_args()
    s = ExpertStateGenerator(args.out_csv_file, args.pgn_file)
    s.generate(write=True)
