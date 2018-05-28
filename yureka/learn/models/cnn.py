import torch.nn as nn
import torch.nn.init as init

from ..data.move_translator import NUM_MOVE_PLANES


class Policy(nn.Module):
    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        hidden_conv_layers,
        batch_norm=False
    ):
        super(Policy, self).__init__()
        self.name = name
        self.batch_norm = batch_norm
        self.conv1 = self.create_conv_layer(
            in_channels,
            out_channels,
            5,
            padding=2
        )
        self.hidden_conv_layers = nn.Sequential(*(self.create_conv_layer(
            out_channels,
            out_channels,
            3,
            padding=1
        ) for _ in range(hidden_conv_layers)))
        self.final_conv_layer = self.create_conv_layer(
            out_channels,
            NUM_MOVE_PLANES,
            1
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
            elif type(m) == nn.BatchNorm2d:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def create_conv_layer(self, *args, **kwargs):
        layers = [nn.Conv2d(*args, **kwargs)]
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(args[1]))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        # x.shape = (batch_size, in_channels, 8, 8)
        x = self.conv1(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        x = self.hidden_conv_layers(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        x = self.final_conv_layer(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        return x


class Value(Policy):
    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        hidden_conv_layers,
        linear_relu_out,
        batch_norm=False
    ):
        super(Value, self).__init__(
            name,
            in_channels,
            out_channels,
            hidden_conv_layers,
            batch_norm=batch_norm
        )
        self.linear_relu = nn.Sequential(
            nn.Linear(NUM_MOVE_PLANES * 8 * 8, linear_relu_out),
            nn.ReLU()
        )
        self.linear_tanh = nn.Sequential(
            nn.Linear(linear_relu_out, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = super(Value, self).forward(x)
        # x.shape = (batch_size, out_channels, 8, 8)
        x = self.linear_relu(x.view(x.shape[0], -1))
        # x.shape = (batch_size, linear_relu_out)
        x = self.linear_tanh(x)
        # x.shape = (batch_size, 1)
        return x


models = {
    'Policy.v0': {
        'class': Policy,
        'args': (23, 128, 11),
        'kwargs': {},
    },
    'Policy.v1': {
        'class': Policy,
        'args': (23, 128, 20),
        'kwargs': {
            'batch_norm': True,
        },
    },
    'Policy.v2': {
        'class': Policy,
        'args': (119, 256, 11),
        'kwargs': {
            'batch_norm': True,
        },
    },
    'Value.v0': {
        'class': Value,
        'args': (23, 128, 11, 128),
        'kwargs': {},
    },
    'Value.v1': {
        'class': Value,
        'args': (23, 128, 20, 128),
        'kwargs': {
            'batch_norm': True,
        },
    },
    'Value.v2': {
        'class': Value,
        'args': (119, 256, 11, 256),
        'kwargs': {
            'batch_norm': True,
        },
    },
    'Rollout.v0': {
        'class': Policy,
        'args': (23, 128, 3),
        'kwargs': {},
    },
    'Rollout.v1': {
        'class': Policy,
        'args': (23, 64, 1),
        'kwargs': {},
    },
}


def create(model_name):
    model_setting = models[model_name]
    return model_setting['class'](
        model_name,
        *model_setting['args'],
        **model_setting['kwargs'],
    )
