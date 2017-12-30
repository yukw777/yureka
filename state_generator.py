import attr
import collections
import chess
import chess.pgn
import pandas as pd


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


@attr.s
class StateGenerator():
    game_file_name = attr.ib()

    def __attrs_post_init__(self):
        self.game_file = open(self.game_file_name, 'r')

    def get_game(self):
        while True:
            g = chess.pgn.read_game(self.game_file)
            if g is None:
                break
            yield g

    def get_square_piece_value(self, piece_map, square, piece):
        p = piece_map.get(square)
        if p and p == piece:
            return 1
        else:
            return 0

    def get_square_piece_data(self, game):
        board = game.board()
        for move in game.main_line():
            piece_map = board.piece_map()
            data_dict = {}
            for sq, sq_name in enumerate(chess.SQUARE_NAMES):
                for piece in pieces:
                    val = self.get_square_piece_value(piece_map, sq, piece)
                    data_dict[f'{sq_name}-{piece.symbol()}'] = val
            yield data_dict
            board.push(move)

    def get_repetition_data(self, game):
        board = game.board()
        transpositions = collections.Counter()
        for move in game.main_line():
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
            yield data_dict
            board.push(move)

    def get_turn_data(self, game):
        board = game.board()
        for move in game.main_line():
            yield {'turn': 1 if board.turn else 0}  # 1 if white else 0
            board.push(move)

    def get_move_count_data(self, game):
        board = game.board()
        for move in game.main_line():
            yield {'move_count': board.fullmove_number}
            board.push(move)

    def generate(self):
        df = pd.DataFrame()
        for game in self.get_game():
            sq_piece_df = pd.DataFrame(self.get_square_piece_data(game))
            df = pd.concat([df, sq_piece_df])

        return df


if __name__ == '__main__':
    s = StateGenerator('tests/test.pgn')
    df = s.generate()
    print(df.head())
    print(df.shape)
