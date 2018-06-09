import torch.nn as nn


class MSECrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MSECrossEntropyLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        pred_move, pred_value = outputs
        move, value = labels
        cross_entropy_loss = self.cross_entropy(pred_move, move)
        mse_loss = self.mse(pred_value, value)
        return (
            0.01 * mse_loss + cross_entropy_loss,
            mse_loss,
            cross_entropy_loss,
        )
