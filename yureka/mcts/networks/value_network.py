import attr
import chess
import torch

from ...learn.data.board_data import get_board_data
from ...learn.data.chess_dataset import get_tensor_from_row


@attr.s
class ValueNetwork():
    network = attr.ib()
    cuda = attr.ib(default=True)
    cuda_device = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.network.eval()
        self.cuda = self.cuda and torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device('cuda', self.cuda_device)
        else:
            self.device = torch.device('cpu')
        self.network.to(self.device)

    def get_value(self, board, color):
        board_data = get_board_data(board, color)
        with torch.no_grad():
            tensor = get_tensor_from_row(board_data)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)
            value = self.network(tensor).squeeze().item()

            # value network returns the result in the perspective of
            # WHITE. So, we need to negate it if color is black
            if color == chess.BLACK:
                return -value
            return value
