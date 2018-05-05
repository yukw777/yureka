import attr
import torch

from torch.autograd import Variable

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
            self.network.cuda(self.cuda_device)

    def get_value(self, board):
        board_data = get_board_data(board)
        tensor = get_tensor_from_row(board_data)
        tensor = tensor.unsqueeze(0)
        value = self.network(Variable(tensor.cuda(), volatile=True))
        return value.squeeze().data[0]
