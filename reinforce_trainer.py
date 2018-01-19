import sys
import importlib
import torch.multiprocessing as mp
import multiprocessing
multiprocessing.Queue = mp.Queue
multiprocessing.Process = mp.Process
multiprocessing.SimpleQueue = mp.SimpleQueue
multiprocessing.Pool = mp.Pool
sys.modules['multiprocessing.reduction'] = importlib.import_module(
    'torch.multiprocessing.reductions')
import attr
import os
import chess
import torch
import datetime
import logging
import random
import threading
import glob
import models
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.optim as optim
from chess_engine import ChessEngine


@attr.s
class ReinforceTrainer():
    model = attr.ib()
    opponent_pool_path = attr.ib()
    trainee_saved_model = attr.ib()
    learning_rate = attr.ib(default=1e-4)
    num_iter = attr.ib(default=10000)
    num_games = attr.ib(default=64)
    log_interval = attr.ib(default=10)
    save_interval = attr.ib(default=500)
    multi_threaded = attr.ib(default=True)
    logger = attr.ib(default=logging.getLogger(__name__))

    def __attrs_post_init__(self):
        self.trainee_model = models.create(self.model)
        if self.trainee_saved_model:
            self.trainee_model.load_state_dict(
                torch.load(self.trainee_saved_model))
        if self.multi_threaded:
            mp.set_start_method('spawn')
            self.trainee_model.share_memory()
            # self.lock = threading.Lock()

    def self_play_log(self, color, reward, policy_loss):
        str_color = "white" if color == chess.WHITE else "black"
        self.logger.debug(f'Trainee color: {str_color}\tReward: {reward}\t'
                          f'Policy loss: {policy_loss.data[0]}')

    def get_opponent(self):
        # NOTE: We need to lock it when creating a new model b/c of a bug
        # https://github.com/pytorch/pytorch/issues/1868
        # if self.multi_threaded:
            # with self.lock:
                # opponent_model = models.create(self.model)
        # else:
        opponent_model = models.create(self.model)

        opponent_model_files = glob.glob(os.path.join(
            self.opponent_pool_path, '*.model'))
        opponent_model_file = random.choice(opponent_model_files)
        opponent_model.load_state_dict(torch.load(opponent_model_file))
        if self.multi_threaded:
            opponent_model.share_memory()
        return ChessEngine(opponent_model, train=False)

    def game(self, number):
        self.logger.debug(f'Staring game {number}')
        trainee_color = random.choice([chess.WHITE, chess.BLACK])
        trainee_engine = ChessEngine(self.trainee_model)
        color, reward, policy_loss = self_play(
            trainee_color, trainee_engine, self.get_opponent())
        self.self_play_log(color, reward, policy_loss)
        return policy_loss

    def setup_games(self):
        color = random.choice([chess.WHITE, chess.BLACK])
        trainee = ChessEngine(self.trainee_model)
        opponent = self.get_opponent()
        return color, trainee, opponent

    def collect_policy_losses(self):
        if self.multi_threaded:
            policy_losses = []
            with ProcessPoolExecutor() as executor:
                game_futures = [executor.submit(self_play, *self.setup_games())
                                for _ in range(self.num_games)]
                for future in as_completed(game_futures):
                    color, reward, policy_loss = future.result()
                    self.self_play_log(color, reward, policy_loss)
                    policy_losses.append(policy_loss)
            return policy_losses
            # policy_losses = []
            # with mp.Pool() as p:
                # for color, reward, policy_loss in p.starmap(
                    # self_play, [self.setup_games() for _ in
                    # range(self.num_games)]):
                    # self.self_play_log(color, reward, policy_loss)
                    # policy_losses.append(policy_loss)
            # return policy_losses
        else:
            return [self.game(n) for n in range(self.num_games)]

    def run(self):
        self.logger.info('Training starting...')
        self.logger.info(f'Model: {self.model}')
        self.logger.info(f'Opponent pool: {self.opponent_pool_path}')
        self.logger.info(f'Trainee saved model: {self.trainee_saved_model}')
        self.logger.info(f'Learning rate: {self.learning_rate}')
        self.logger.info(f'Number of iterations: {self.num_iter}')
        self.logger.info(f'Number of games: {self.num_games}')
        self.logger.info(f'Log interval: {self.log_interval}')
        self.logger.info(f'Save interval: {self.save_interval}')
        self.logger.info(f'Multi threaded: {self.multi_threaded}')

        optimizer = optim.SGD(
            self.trainee_model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True
        )
        for i in range(self.num_iter):
            policy_losses = self.collect_policy_losses()
            import pdb; pdb.set_trace()
            optimizer.zero_grad()
            policy_loss = torch.cat(policy_losses).sum()
            policy_loss /= self.num_games
            msg = f'Total policy loss for iteration {i}: {policy_loss.data[0]}'
            if i % self.log_interval == self.log_interval - 1:
                self.logger.info(msg)
            else:
                self.logger.debug(msg)

            policy_loss.backward()
            optimizer.step()
            if i != 0 and i % self.save_interval == 0:
                self.save(i)
        self.logger.info('Training done')

    def save(self, iteration):
        filename = self.trainee_model.__class__.__name__
        filename += f"_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}"
        filename += f"_{iteration}.model"
        filepath = os.path.join(
            os.getcwd(),
            self.opponent_pool_path,
            filename
        )
        self.logger.info(f'Saving: {filepath}')
        torch.save(self.trainee_model.state_dict(), filepath)
        self.logger.info('Done saving')


def get_reward(result, color):
    points = result.split('-')
    if color == chess.WHITE:
        player_point = points[0]
    else:
        player_point = points[1]

    if player_point == '0':
        return -1
    elif player_point == '1/2':
        return 0
    elif player_point == '1':
        return 1
    else:
        raise Exception(f'Unknown result: {result}, {color}')


def self_play(color, trainee, opponent):
    log_probs = []
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if board.turn == color:
            move, log_prob = trainee.get_move(board)
            log_probs.append(log_prob)
        else:
            move = opponent.get_move(board)
        board.push(move)

    # TODO: set baseline with the value network
    baseline = 0
    result = board.result(claim_draw=True)
    reward = get_reward(result, color)
    policy_loss = -torch.cat(log_probs).sum() * (reward - baseline)
    return color, reward, policy_loss


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('opponent_pool_path')
    parser.add_argument('trainee_saved_model')
    parser.add_argument('-r', '--learning-rate', type=float)
    parser.add_argument('-i', '--num-iter', type=int)
    parser.add_argument('-g', '--num-games', type=int)
    parser.add_argument('-l', '--log-file')
    parser.add_argument('-s', '--save-interval', type=int)
    parser.add_argument('-o', '--log-interval', type=int)
    parser.add_argument('-t', '--single-threaded', action="store_true")
    parser.add_argument('-d', '--debug', action="store_true")

    args = parser.parse_args()

    logger = logging.getLogger('ReinforceTrainer')
    logging_config = {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'level': logging.DEBUG if args.debug else logging.INFO,
    }
    if args.log_file:
        logging_config['filename'] = args.log_file
    logging.basicConfig(**logging_config)

    trainer_setting = {
        'model': args.model,
        'opponent_pool_path': args.opponent_pool_path,
        'trainee_saved_model': args.trainee_saved_model,
        'logger': logger,
    }
    if args.learning_rate:
        trainer_setting['learning_rate'] = args.learning_rate
    if args.num_iter:
        trainer_setting['num_iter'] = args.num_iter
    if args.num_games:
        trainer_setting['num_games'] = args.num_games
    if args.save_interval:
        trainer_setting['save_interval'] = args.save_interval
    if args.log_interval:
        trainer_setting['log_interval'] = args.log_interval
    if args.single_threaded:
        trainer_setting['multi_threaded'] = not args.single_threaded

    trainer = ReinforceTrainer(**trainer_setting)
    trainer.run()


if __name__ == '__main__':
    run()
