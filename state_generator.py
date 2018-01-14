import attr
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


@attr.s
class StateGenerator():
    game_file_name = attr.ib()
    out_csv_file = attr.ib()

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
            return True
        else:
            return False

    def get_square_piece_data(self, game):
        board = game.board()
        for move in game.main_line():
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
                    occupied = self.get_square_piece_value(
                        piece_map, sq, piece)

                    key = f'{sq_name_for_player}-{piece.symbol()}'
                    if occupied:
                        if piece.color == chess.WHITE:
                            white_data.append(key)
                        else:
                            black_data.append(key)
            yield {
                'white_square_piece': ','.join(white_data),
                'black_square_piece': ','.join(black_data),
            }
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

    def get_castling_data(self, game):
        board = game.board()
        for move in game.main_line():
            w_kingside = 1 if board.has_kingside_castling_rights(
                chess.WHITE) else 0
            w_queenside = 1 if board.has_queenside_castling_rights(
                chess.WHITE) else 0
            b_kingside = 1 if board.has_kingside_castling_rights(
                chess.BLACK) else 0
            b_queenside = 1 if board.has_queenside_castling_rights(
                chess.BLACK) else 0
            yield {
                'w_kingside_castling': w_kingside,
                'w_queenside_castling': w_queenside,
                'b_kingside_castling': b_kingside,
                'b_queenside_castling': b_queenside,
            }
            board.push(move)

    def get_no_progress_data(self, game):
        board = game.board()
        for move in game.main_line():
            yield {'no_progress': int(board.halfmove_clock / 2)}
            board.push(move)

    def get_move_data(self, game):
        board = game.board()
        for move in game.main_line():
            yield {'move': move_translator.translate_to_engine_move(
                move, board.turn)}
            board.push(move)

    def generate(self, write=False):
        count = 0
        df = pd.DataFrame()
        header = True
        for game in self.get_game():
            count += 1
            combined_df = pd.concat([
                pd.DataFrame(self.get_square_piece_data(game)),
                pd.DataFrame(self.get_repetition_data(game)),
                pd.DataFrame(self.get_turn_data(game)),
                pd.DataFrame(self.get_move_count_data(game)),
                pd.DataFrame(self.get_castling_data(game)),
                pd.DataFrame(self.get_no_progress_data(game)),
            ], axis=1)
            combined_df = pd.concat([
                combined_df,
                pd.DataFrame(self.get_move_data(game))
            ], axis=1)
            df = pd.concat([df, combined_df])
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pgn_file')
    parser.add_argument('out_csv_file')
    args = parser.parse_args()
    s = StateGenerator(args.pgn_file, args.out_csv_file)
    s.generate(write=True)
