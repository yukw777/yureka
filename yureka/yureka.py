#!/home/keunwoo/Documents/Projects/chess-engine/venv/bin/python

from .engine import UCIMCTSEngine
from .engine.utils import print_flush

if __name__ == '__main__':
    print_flush('Yureka!')
    UCIMCTSEngine().listen()
