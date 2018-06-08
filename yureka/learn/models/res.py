import torch.nn as nn

from ..data.move_translator import NUM_MOVE_PLANES


class ConvBlock(nn.Module):
    def __init__(self, filters, *args, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            *(nn.Conv2d(*args, **kwargs) for _ in range(filters)))
        self.batch_norm = nn.BatchNorm2d(args[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, filters, *args, **kwargs):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            *(nn.Conv2d(*args, **kwargs) for _ in range(filters)))
        self.batch_norm1 = nn.BatchNorm2d(args[1])
        self.relu = nn.ReLU()
        self.conv2 = nn.Sequential(
            *(nn.Conv2d(*args, **kwargs) for _ in range(filters)))
        self.batch_norm2 = nn.BatchNorm2d(args[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        return self.relu(out + x)


class PolicyHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PolicyHead, self).__init__()
        self.conv = nn.Sequential(
            *(nn.Conv2d(*args, 1, **kwargs) for _ in range(2)))
        self.batch_norm = nn.BatchNorm2d(args[1])
        self.relu = nn.ReLU()
        self.linear = nn.Linear(args[1], NUM_MOVE_PLANES)

    def forward(self, x):
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.conv(x)
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.batch_norm(x)
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.relu(x)
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.linear(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(*args, 1, **kwargs)
        self.batch_norm1 = nn.BatchNorm2d(args[1])
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(args[1], hidden_size)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        x = self.tanh(x)
        return x


class ResNet(nn.Module):
    def __init__(self, name, tower, head):
        super(ResNet, self).__init__()
        self.name = name
        self.tower = tower
        self.head = head

    def forward(self, x):
        x = self.tower(x)
        return self.head(x)
