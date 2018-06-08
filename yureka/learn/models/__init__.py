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

resnet_setting = {
    'ResNet.v1': {
        'in_channels': 8,
        'out_channels': 8,
        'conv_block_filters': 24,
        'conv_block_kernel': 3,
        'conv_block_padding': 1,
        'conv_block_stride': 1,
        'res_block_filters': 24,
        'res_block_kernel': 3,
        'res_block_padding': 1,
        'res_block_stride': 1,
        'res_blocks': 9,
    },
}


def create(model_name):
    model_setting = models[model_name]
    return model_setting['class'](
        model_name,
        *model_setting['args'],
        **model_setting['kwargs'],
    )
