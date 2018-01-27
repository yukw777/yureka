import attr
import chess
import pandas as pd
import numpy as np
import torch
import move_translator
from board_data import BOARD_SIZE
from torch.utils.data import Dataset


SIZE = (1, ) + BOARD_SIZE


@attr.s
class ChessDataset(Dataset):
    data_file = attr.ib()
    label_name = attr.ib(default='move')

    def __attrs_post_init__(self):
        self.df = pd.read_csv(self.data_file)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (
            get_tensor_from_row(row),
            move_translator.get_engine_move_index(row[self.label_name]),
        )


def get_tensor_from_row(row):
    return torch.from_numpy(np.vstack((
        get_board_data(row, bool(row['turn'])),
        np.full(SIZE, row['turn']),
        np.full(SIZE, row['move_count']),
        np.full(SIZE, row['b_kingside_castling']),
        np.full(SIZE, row['b_queenside_castling']),
        np.full(SIZE, row['w_kingside_castling']),
        np.full(SIZE, row['w_queenside_castling']),
        np.full(SIZE, row['no_progress']),
        np.full(SIZE, 0),
    ))).float()


def get_square_piece_data(data):
    board_data = np.full(
        (len(chess.PIECE_TYPES), BOARD_SIZE[0] * BOARD_SIZE[1]), 0)

    for sq_symbol in data.split(','):
        sq, symbol = sq_symbol.split('-')
        piece = chess.Piece.from_symbol(symbol)
        square = move_translator.square_name_to_square(sq)
        board_data[piece.piece_type - chess.PAWN][square] = 1

    return board_data.reshape((len(chess.PIECE_TYPES), ) + BOARD_SIZE)


def get_board_data(row, turn):
    white_data = get_square_piece_data(row['white_square_piece'])
    black_data = get_square_piece_data(row['black_square_piece'])
    if turn == chess.WHITE:
        piece_data = np.vstack((white_data, black_data))
    else:
        piece_data = np.vstack((black_data, white_data))

    return np.vstack((
        piece_data,
        np.full(SIZE, 1),
        np.full(SIZE, row['rep_2']),
        np.full(SIZE, row['rep_3']),
    ))
