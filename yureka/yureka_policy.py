import os

from .engine.constants import DEFAULT_MODEL, DEFAULT_MODEL_FILE
from .engine import UCIPolicyEngine
from .chess_engine import print_flush


if __name__ == '__main__':
    import argparse
    default_model = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'saved_models',
        'RL',
        'ChessEngine_2018-01-23_19:22:59_1000.model',
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL)
    parser.add_argument('-f', '--model-file', default=DEFAULT_MODEL_FILE)
    parser.add_argument('-c', '--cuda-device', type=int)

    args = parser.parse_args()
    print_flush('Yureka!')
    uci = UCIPolicyEngine(
        model_name=args.model,
        model_file=os.path.expanduser(args.model_file),
        cuda_device=args.cuda_device
    )
    uci.listen()
