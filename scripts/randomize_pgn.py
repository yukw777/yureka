import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('in_pgn')
parser.add_argument('out_pgn')
args = parser.parse_args()

with open(args.out_pgn, 'w') as out, open(args.in_pgn, 'r') as in_file:
    pgn = in_file.read()
    games = pgn.split('\n\n[')
    out.write('[' + '\n\n['.join(sorted(games, key=lambda x: random.random())))
