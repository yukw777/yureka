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
        'args': (119, 256, 23, 256),
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
