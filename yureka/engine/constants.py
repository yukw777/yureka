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
TC_OPPONENT_TIME_RATIO = 0.5
TC_SUDDEN_DEATH_THRESHOLD = 30000  # 30 seconds left

root_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# UCI Policy Engine
DEFAULT_MODEL = 'Policy.v2'
DEFAULT_MODEL_FILE = os.path.join(
    root_path,
    'saved_models',
    'Policy.v2',
    'Policy.v2_2018-06-19_21_08_58_19.model',
)

# UCI MCTS Engine
RANDOM_POLICY = 'random'
DEFAULT_VALUE = 'Value.v2'
DEFAULT_VALUE_FILE = os.path.join(
    root_path,
    'saved_models',
    'Value.v2',
    'Value.v2_2018-06-19_18_16_43_18.model',
)
ZERO_VALUE = 'zero'
DEFAULT_POLICY = 'Policy.v2'
DEFAULT_POLICY_FILE = os.path.join(
    root_path,
    'saved_models',
    'Policy.v2',
    'Policy.v2_2018-06-19_21_08_58_19.model',
)
DEFAULT_RESNET = 'ResNet.v1'
DEFAULT_RESNET_TOWER_FILE = os.path.join(
    root_path,
    'saved_models',
    'ResNet.v1',
    'Tower_2018-06-26_07_17_17_10.model',
)
DEFAULT_RESNET_VALUE_FILE = os.path.join(
    root_path,
    'saved_models',
    'ResNet.v1',
    'ValueHead_2018-06-26_17_36_08_11.model',
)
DEFAULT_RESNET_POLICY_FILE = os.path.join(
    root_path,
    'saved_models',
    'ResNet.v1',
    'PolicyHead_2018-06-26_17_36_08_11.model',
)
