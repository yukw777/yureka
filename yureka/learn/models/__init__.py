import torch.nn as nn

from . import res
from .cnn import Policy, Value


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
        'args': (21, 128, 11),
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
        'args': (21, 128, 11, 128),
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

resnet_settings = {
    'ResNet.v0': {
        'in_channels': 21,
        'out_channels': 21,
        'conv_block_filters': 3,
        'conv_block_kernel': 3,
        'conv_block_padding': 1,
        'conv_block_stride': 1,
        'res_block_filters': 3,
        'res_block_kernel': 3,
        'res_block_padding': 1,
        'res_block_stride': 1,
        'res_blocks': 6,
        'value_hidden': 24,
    },
}


def create_res(model_name):
    setting = resnet_settings[model_name]
    tower = [
        res.ConvBlock(
            setting['conv_block_filters'],
            setting['in_channels'],
            setting['out_channels'],
            setting['conv_block_kernel'],
            padding=setting['conv_block_padding'],
            stride=setting['conv_block_stride']
        ),
    ]
    tower += [
        res.ResBlock(
            setting['res_block_filters'],
            setting['in_channels'],
            setting['out_channels'],
            setting['res_block_kernel'],
            padding=setting['res_block_padding'],
            stride=setting['res_block_stride']
        ) for _ in range(setting['res_blocks'])
    ]
    tower = nn.Sequential(*tower)

    policy = res.PolicyHead(
        setting['in_channels'],
        setting['out_channels']
    )
    value = res.ValueHead(
        setting['value_hidden'],
        setting['in_channels'],
        setting['out_channels']
    )

    return tower, policy, value


def create(model_name):
    model_setting = models[model_name]
    return model_setting['class'](
        model_name,
        *model_setting['args'],
        **model_setting['kwargs'],
    )
