import os


# Time control
TC_WTIME = 'wtime'
TC_BTIME = 'btime'
TC_WINC = 'winc'
TC_BINC = 'binc'
TC_MOVESTOGO = 'movestogo'
TC_MOVETIME = 'movetime'
TC_KEYS = [
    TC_WTIME,
    TC_BTIME,
    TC_WINC,
    TC_BINC,
    TC_MOVESTOGO,
    TC_MOVETIME,
]

# UCI Policy Engine
DEFAULT_MODEL = 'Policy.v0'
DEFAULT_MODEL_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'saved_models',
    'SL_endgame',
    'Policy_2018-01-27_07:09:34_14.model',
)

# UCI MCTS Engine
root_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
RANDOM_POLICY = 'random'
DEFAULT_VALUE = 'Value.v0'
DEFAULT_VALUE_FILE = os.path.join(
    root_path,
    'saved_models',
    'Value',
    'Value_2018-01-31_14:20:50_4.model',
)
ZERO_VALUE = 'zero'
DEFAULT_POLICY = 'Policy.v0'
DEFAULT_POLICY_FILE = os.path.join(
    root_path,
    'saved_models',
    'SL_endgame',
    'Policy_2018-01-27_07:09:34_14.model',
)
DEFAULT_RESNET = 'ResNet.v0'
DEFAULT_RESNET_TOWER_FILE = ''
DEFAULT_RESNET_VALUE_FILE = ''
DEFAULT_RESNET_POLICY_FILE = ''
