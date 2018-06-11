import attr
import random
import chess
import chess.pgn
import pandas as pd
import sys
import torch
import lmdb
import os

from ...mcts.networks import PolicyNetwork
from ..models import cnn

from . import move_translator
from .board_data import get_reward, get_board_data


@attr.s
class StateGenerator():
    out_file_name = attr.ib()
    out_file_type = attr.ib()
    history = attr.ib()

    def __attrs_post_init__(self):
        if self.out_file_type == 'csv':
            self.print_header = True
        elif self.out_file_type == 'lmdb':
            if not os.path.exists(self.out_file_name):
                os.makedirs(self.out_file_name)
            self.env = lmdb.open(self.out_file_name, map_size=2e11)
            self.txn = self.env.begin(write=True)
            self.cursor = self.txn.cursor()

    def __del__(self):
        if self.out_file_type == 'lmdb':
            self.cursor.close()
            self.txn.commit()
            self.env.close()

    def get_game(self):
        raise NotImplemented

    def get_label_data(self):
        raise NotImplemented

    def get_game_data(self, game):
        raise NotImplemented

    def stop(self, game_count, state_count):
        pass

    def write(self, df, state_count):
        if self.out_file_type == 'csv':
            return self.write_csv(df, state_count)
        elif self.out_file_type == 'lmdb':
            return self.write_lmdb(df, state_count)

    def write_csv(self, df, state_count):
        df.to_csv(
            self.out_file_name,
            index=False,
            header=self.print_header,
            mode='a'
        )
        self.print_header = False
        return state_count + df.shape[0]

    def write_lmdb(self, df, state_count):
        print(f'writing from id {state_count}')
        items = []
        # iterate over rows in random order
        for _, row in df.sample(frac=1).iterrows():
            items.append((f'{state_count}'.encode(), row.to_msgpack()))
            state_count += 1
        consumed, added = self.cursor.putmulti(items)
        self.txn.commit()
        self.cursor.close()
        print(f'{consumed} rows consumed, {added} rows added')
        self.txn = self.env.begin(write=True)
        self.cursor = self.txn.cursor()
        return state_count

    def generate(self, skip=None, write=False):
        count = 0
        state_count = 0
        df = pd.DataFrame()
        print(f'skipping: {skip}')
        for game in self.get_game():
            count += 1
            if skip and count <= skip:
                if count % 100 == 0:
                    print(f'Skipped {count}')
                continue
            try:
                game_df = pd.DataFrame(self.get_game_data(game))
                game_df = pd.concat([
                    game_df,
                    pd.DataFrame(self.get_label_data(game))
                ], axis=1)
            except Exception as e:
                # just catch and move on to the next one
                print(e)
                continue
            df = pd.concat([df, game_df])
            if count % 100 == 0:
                if write:
                    state_count = self.write(df, state_count)
                    df = pd.DataFrame()
                else:
                    state_count = df.shape[0]
                print(f'{count} games processed...')
                print(f'{state_count} states generated...')
            self.stop(count, state_count)
        if write:
            self.write(df, state_count)

        return df


@attr.s
class SampledStateGenerator(StateGenerator):
    both_colors = attr.ib()

    def get_game_data(self, data):
        board, _, _, _ = data
        game_df = get_board_data(board, board.turn, self.history)
        if self.both_colors:
            opposite = get_board_data(
                board, not board.turn, self.history)
            return [game_df, opposite]
        else:
            return [game_df]

    def get_label_data(self, data):
        board, move, reward, opposite_reward = data
        if self.both_colors:
            return [{
                'value': reward,
                'move': move_translator.translate_to_engine_move(
                    move, board.turn)
            }, {
                'value': opposite_reward,
                'move': move_translator.translate_to_engine_move(
                    move, not board.turn)
            }]
        else:
            return [{
                'value': reward,
                'move': move_translator.translate_to_engine_move(
                    move, board.turn)
            }]


@attr.s
class SimSampledStateGenerator(SampledStateGenerator):
    sl_engine = attr.ib()
    rl_engine = attr.ib()
    num_games = attr.ib()

    def get_game(self):
        for i in range(self.num_games):
            while True:
                # based on this statistics
                # https://chess.stackexchange.com/questions/2506/what-is-the-average-length-of-a-game-of-chess
                sampled = random.randint(1, 100)
                board = chess.Board()
                t = 0
                while not board.is_game_over(claim_draw=True):
                    if t < sampled:
                        move, _ = self.sl_engine.get_move(board)
                    elif t == sampled:
                        move = random.choice(list(board.legal_moves))
                        # NOTE: Be careful! The last move on sampled_board
                        # is not the one made by color!
                        sampled_move = move
                        color = board.turn
                        sampled_board = board.copy()
                    else:
                        move, _ = self.rl_engine.get_move(board)
                    board.push(move)
                    t += 1
                if t <= sampled:
                    print(f'We drew {sampled} state '
                          f'but the game only got to {t}')
                    print("Let's try again")
                else:
                    break
            result = board.result(claim_draw=True)
            reward = get_reward(result, color)
            opposite_reward = get_reward(result, not color)
            yield sampled_board, sampled_move, reward, opposite_reward


@attr.s
class ExpertStateGenerator(StateGenerator):
    game_file_name = attr.ib()
    num_states = attr.ib()

    def __attrs_post_init__(self):
        super(ExpertStateGenerator, self).__attrs_post_init__()
        self.game_file = open(self.game_file_name, 'r')

    def get_game(self):
        while True:
            g = chess.pgn.read_game(self.game_file)
            if g is None:
                break
            yield g

    def get_game_data(self, game):
        board = game.board()
        for move in game.main_line():
            yield get_board_data(board, board.turn, self.history)
            board.push(move)

    def get_label_data(self, game):
        board = game.board()
        result = game.headers['Result']
        # calculate the reward from WHITE's perspective.
        reward = get_reward(result, chess.WHITE)
        for move in game.main_line():
            yield {
                'move': move_translator.translate_to_engine_move(
                    move, board.turn),
                'value': reward,
            }
            board.push(move)

    def stop(self, game_count, state_count):
        if state_count > self.num_states:
            print('Generated enough states. Exiting...')
            sys.exit()


@attr.s
class ExpertSampledStateGenerator(SampledStateGenerator, ExpertStateGenerator):

    def get_game(self):
        for game in ExpertStateGenerator.get_game(self):
            moves = list(game.main_line())
            try:
                # leave at least one move
                sampled = random.randint(1, len(moves) - 1)
            except ValueError as e:
                print(e)
                continue
            board = chess.Board()
            color = None
            for i in range(sampled):
                board.push(moves[i])
            # NOTE: Be careful! The last move on sampled_board
            # is not the one made by color!
            color = board.turn
            # have to get the result from the headers b/c people resign
            result = game.headers['Result']
            try:
                reward = get_reward(result, color)
                opposite_reward = get_reward(result, not color)
            except Exception as e:
                print(e)
                continue
            yield board, moves[sampled], reward, opposite_reward

    def stop(self, game_count, state_count):
        ExpertStateGenerator.stop(self, game_count, state_count)


def expert(args):
    s = ExpertStateGenerator(
        args.out_file_name,
        args.format,
        args.history,
        args.pgn_file,
        args.num_states
    )
    s.generate(write=True, skip=args.skip)


def sim_sampled(args):
    sl = cnn.create(args.sl_engine_name)
    sl.load_state_dict(torch.load(args.sl_engine_file))
    sl = PolicyNetwork(sl)
    rl = cnn.create(args.rl_engine_name)
    rl.load_state_dict(torch.load(args.rl_engine_file))
    rl = PolicyNetwork(rl)
    u = SimSampledStateGenerator(
        args.out_file_name,
        args.format,
        args.history,
        args.both_color,
        sl,
        rl,
        args.num_games
    )
    u.generate(write=True)


def expert_sampled(args):
    s = ExpertSampledStateGenerator(
        args.pgn_file,
        args.num_states,
        args.out_file_name,
        args.format,
        args.history,
        args.both_colors
    )
    s.generate(write=True, skip=args.skip)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_expert = subparsers.add_parser('expert')
    parser_expert.add_argument('pgn_file')
    parser_expert.add_argument('out_file_name')
    parser_expert.add_argument('num_states', type=int)
    parser_expert.add_argument('-s', '--skip', type=int)
    parser_expert.add_argument('--history', type=int, default=1)
    parser_expert.add_argument('-f', '--format', default='csv')
    parser_expert.set_defaults(func=expert)

    parser_sim_sampled = subparsers.add_parser('sim_sampled')
    parser_sim_sampled.add_argument('sl_engine_name')
    parser_sim_sampled.add_argument('sl_engine_file')
    parser_sim_sampled.add_argument('rl_engine_name')
    parser_sim_sampled.add_argument('rl_engine_file')
    parser_sim_sampled.add_argument('num_games', type=int)
    parser_sim_sampled.add_argument('out_file_name')
    parser_sim_sampled.add_argument('-b', '--both-colors', action='store_true')
    parser_sim_sampled.add_argument('-f', '--format', default='csv')
    parser_sim_sampled.add_argument('--history', type=int, default=1)
    parser_sim_sampled.set_defaults(func=sim_sampled)

    parser_expert_sampled = subparsers.add_parser('expert_sampled')
    parser_expert_sampled.add_argument('pgn_file')
    parser_expert_sampled.add_argument('out_file_name')
    parser_expert_sampled.add_argument('num_states', type=int)
    parser_expert_sampled.add_argument('-s', '--skip', type=int)
    parser_expert_sampled.add_argument('-f', '--format', default='csv')
    parser_expert_sampled.add_argument('--history', type=int, default=1)
    parser_expert_sampled.add_argument(
        '-b', '--both-colors', action='store_true')
    parser_expert_sampled.set_defaults(func=expert_sampled)
    args = parser.parse_args()
    args.func(args)
