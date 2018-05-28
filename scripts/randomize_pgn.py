import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('in_pgn')
parser.add_argument('out_pgn')
parser.add_argument('-e', '--encoding')
args = parser.parse_args()

with open(args.out_pgn, 'w', encoding='utf-8') as out, \
     open(args.in_pgn, 'r', encoding=args.encoding) as in_file:
    pgn = in_file.read()
    games = pgn.split('\n\n[')
    out.write('[' + '\n\n['.join(sorted(games, key=lambda x: random.random())))
