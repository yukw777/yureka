import attr
import chess
import pandas as pd
import numpy as np
import torch
import itertools
import lmdb

from torch.utils.data import Dataset

from . import move_translator
from .board_data import BOARD_SIZE


SIZE = (1, ) + BOARD_SIZE


@attr.s
class LMDBChessDataset(Dataset):
    lmdb_name = attr.ib()
    offset = attr.ib(default=0)
    limit = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.env = lmdb.open(self.lmdb_name, map_size=2e11)
        self.txn = self.env.begin()
        self.cursor = self.txn.cursor()

    def __len__(self):
        if self.limit:
            return self.limit - self.offset
        return self.env.stat()['entries'] - self.offset

    def __getitem__(self, index):
        index = index + self.offset
        row = pd.read_msgpack(
            self.cursor.get(f'{index}'.encode()),
            encoding='ascii'
        )
        return data_from_row(row)

    def __del__(self):
        self.cursor.close()
        self.txn.commit()
        self.env.close()


def data_from_row(row):
    value = []
    if 'value' in row:
        value = torch.Tensor([float(row['value'])])
    move = []
    if 'move' in row:
        move = move_translator.get_engine_move_index(row['move'])
    return (
        get_tensor_from_row(row),
        move,
        value,
    )


@attr.s
class ChessDataset(Dataset):
    data_file = attr.ib()

    def __attrs_post_init__(self):
        self.df = pd.read_csv(self.data_file, keep_default_na=False)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return data_from_row(row)


def get_tensor_from_row(row):
    return torch.from_numpy(np.vstack((
        get_board_data(row),
        np.full(SIZE, row['color']),
        np.full(SIZE, row['move_count']),
        np.full(SIZE, row['b_castling']),
        np.full(SIZE, row['w_castling']),
        np.full(SIZE, row['no_progress']),
        np.full(SIZE, 0),
    ))).float()


def get_square_piece_data(data):
    board_data = np.full(
        (len(chess.PIECE_TYPES), BOARD_SIZE[0] * BOARD_SIZE[1]), 0)

    if data:
        for sq_symbol in data.split(','):
            sq, symbol = sq_symbol.split('-')
            piece = chess.Piece.from_symbol(symbol)
            square = move_translator.square_name_to_square(sq)
            board_data[piece.piece_type - chess.PAWN][square] = 1

    return board_data.reshape((len(chess.PIECE_TYPES), ) + BOARD_SIZE)


def get_board_data(row):
    white_data = []
    black_data = []
    rep_2_data = []
    rep_3_data = []
    for i in itertools.count():
        try:
            white_data.append(
                get_square_piece_data(row['white_square_piece_%d' % i]))
            black_data.append(
                get_square_piece_data(row['black_square_piece_%d' % i]))
            rep_2_data.append(np.full(SIZE, row['rep_2_%d' % i]))
            rep_3_data.append(np.full(SIZE, row['rep_3_%d' % i]))
        except KeyError:
            break

    white_data = np.vstack(white_data)
    black_data = np.vstack(black_data)
    rep_2_data = np.vstack(rep_2_data)
    rep_3_data = np.vstack(rep_3_data)

    return np.vstack((
        black_data,
        white_data,
        np.full(SIZE, 1),
        rep_2_data,
        rep_3_data,
    ))
