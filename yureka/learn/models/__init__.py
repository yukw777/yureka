import torch.nn as nn

from . import res
from .cnn import Policy, Value


cnn_settings = {
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
        'dropout': False,
        'in_channels': 21,
        'conv_block_out_channels': 128,
        'conv_block_kernel': 3,
        'conv_block_padding': 1,
        'conv_block_stride': 1,
        'res_block_out_channels': 128,
        'res_block_kernel': 3,
        'res_block_padding': 1,
        'res_block_stride': 1,
        'res_blocks': 5,
        'policy_out_channels': 128,
        'value_hidden': 128,
        'value_out_channels': 1,
    },
    'ResNet.v1': {
        'dropout': True,
        'in_channels': 21,
        'conv_block_out_channels': 256,
        'conv_block_kernel': 3,
        'conv_block_padding': 1,
        'conv_block_stride': 1,
        'res_block_out_channels': 256,
        'res_block_kernel': 3,
        'res_block_padding': 1,
        'res_block_stride': 1,
        'res_blocks': 10,
        'policy_out_channels': 128,
        'value_hidden': 256,
        'value_out_channels': 1,
    },
}


def create(model_name):
    if model_name in cnn_settings:
        model_setting = cnn_settings[model_name]
        return model_setting['class'](
            model_name,
            *model_setting['args'],
            **model_setting['kwargs'],
        )
    elif model_name in resnet_settings:
        setting = resnet_settings[model_name]
        tower = [
            res.ConvBlock(
                setting['in_channels'],
                setting['conv_block_out_channels'],
                setting['conv_block_kernel'],
                padding=setting['conv_block_padding'],
                stride=setting['conv_block_stride']
            ),
        ]
        tower += [
            res.ResBlock(
                setting['conv_block_out_channels'],
                setting['res_block_out_channels'],
                setting['res_block_kernel'],
                padding=setting['res_block_padding'],
                stride=setting['res_block_stride']
            ) for _ in range(setting['res_blocks'])
        ]
        if setting['dropout']:
            tower.append(nn.Dropout2d())
        tower = nn.Sequential(*tower)

        policy = res.PolicyHead(
            setting['res_block_out_channels'],
            setting['policy_out_channels']
        )
        value = res.ValueHead(
            setting['value_hidden'],
            setting['res_block_out_channels'],
            setting['value_out_channels']
        )

        return tower, policy, value
