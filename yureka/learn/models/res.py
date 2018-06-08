import torch.nn as nn

from ..data.move_translator import NUM_MOVE_PLANES


class ConvBlock(nn.Module):
    def __init__(self, filters, *args, **kwargs):
        layers = [nn.Conv2d(*args, **kwargs)
                  for _ in range(filters)]
        layers.append(nn.BatchNorm2d(args[1]))
        layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResBlock(nn.Module):
    def __init__(self, filters, *args, **kwargs):
        layers = [nn.Conv2d(*args, **kwargs)
                  for _ in range(filters)]
        layers.append(nn.BatchNorm2d(args[1]))
        layers.append(nn.ReLU())
        layers += [nn.Conv2d(*args, **kwargs)
                   for _ in range(filters)]
        layers.append(nn.BatchNorm2d(args[1]))
        self.network = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.network(x)
        return self.relu(out.stack(x))


class PolicyHead(nn.Module):
    def __init__(self, *args, **kwargs):
        layers = [nn.Conv2d(*args, **kwargs)
                  for _ in range(2)]
        layers.append(nn.BatchNorm2d(args[1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(args[1]), NUM_MOVE_PLANES)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ValueHead(nn.Modules):
    def __init__(self, hidden_size, *args, **kwargs):
        layers = [nn.Conv2d(*args, **kwargs)]
        layers.append(nn.BatchNorm2d(args[1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(args[1]), hidden_size)
        layers.append(nn.ReLU())
        layers.append(hidden_size, 1)
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResNet(nn.Modules):
    def __init__(self, name, tower, head):
        self.name = name
        self.tower = tower
        self.head = head

    def forward(self, x):
        x = self.tower(x)
        return self.head(x)
