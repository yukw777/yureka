import torch.nn as nn

from ..data.move_translator import NUM_MOVE_PLANES


class ConvBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.batch_norm = nn.BatchNorm2d(args[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.conv(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        x = self.batch_norm(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        x = self.relu(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        return x


class ResBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(*args, **kwargs)
        self.batch_norm1 = nn.BatchNorm2d(args[1])
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(*args, **kwargs)
        self.batch_norm2 = nn.BatchNorm2d(args[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape = (batch_size, in_channels, 8, 8)
        out = self.conv1(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        out = self.batch_norm1(out)
        # x.shape = (batch_size, out_channels, 8, 8)
        out = self.relu(out)
        # x.shape = (batch_size, out_channels, 8, 8)
        out = self.conv2(out)
        # x.shape = (batch_size, out_channels, 8, 8)
        out = self.batch_norm2(out)
        # x.shape = (batch_size, out_channels, 8, 8)
        return self.relu(out + x)


class PolicyHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(*args, 1, **kwargs)
        self.batch_norm = nn.BatchNorm2d(args[1])
        self.relu = nn.ReLU()
        self.linear = nn.Linear(
            args[1] * 8 * 8,
            NUM_MOVE_PLANES * 8 * 8,
        )

    def forward(self, x):
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.conv(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        x = self.batch_norm(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        x = self.relu(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        x = self.linear(x.view(x.shape[0], -1))
        # x.shape = (batch_size, NUM_MOVE_PLANES * 8 * 8)
        return x


class ValueHead(nn.Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(*args, 1, **kwargs)
        self.batch_norm1 = nn.BatchNorm2d(args[1])
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(args[1] * 8 * 8, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.conv(x)
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.batch_norm1(x)
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.relu1(x)
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.relu2(x)
        x = self.linear2(x)
        x = self.tanh(x)
        return x


class ResNet(nn.Module):
    def __init__(self, tower, head):
        super(ResNet, self).__init__()
        self.tower = tower
        self.head = head

    def forward(self, x):
        x = self.tower(x)
        return self.head(x)
