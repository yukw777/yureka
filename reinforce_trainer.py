import torch.multiprocessing as mp
import attr
import os
import chess
import torch
import datetime
import logging
import random
import glob
import models
import numpy as np
import torch.optim as optim
from chess_engine import ChessEngine


@attr.s
class PolicyLossIsNan(Exception):
    log_probs = attr.ib()


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
    multi_process = attr.ib(default=True)
    cuda_device = attr.ib(default=None)
    logger = attr.ib(default=logging.getLogger(__name__))

    def __attrs_post_init__(self):
        self.latest_saved_trainee = self.trainee_saved_model
        self.latest_saved_iteration = 0
        self.init_trainee_model_to_latest()
        self.trainee_model = models.create(self.model)
        self.trainee_model.load_state_dict(
            torch.load(self.trainee_saved_model))
        if self.multi_process:
            mp.set_start_method('spawn')
            self.trainee_model.share_memory()

    def init_trainee_model_to_latest(self):
        self.trainee_model = models.create(self.model)
        self.trainee_model.load_state_dict(
            torch.load(self.latest_saved_trainee))
        self.logger.info(f'Trainee model loaded: {self.latest_saved_trainee}')

    def self_play_log(self, color, reward, policy_loss):
        str_color = "white" if color == chess.WHITE else "black"
        self.logger.debug(f'Trainee color: {str_color}\tReward: {reward}\t'
                          f'Policy loss: {policy_loss.data[0]}')

    def get_opponent_model_file(self):
        opponent_model_files = glob.glob(os.path.join(
            self.opponent_pool_path, '*.model'))
        return random.choice(opponent_model_files)

    def setup_games(self, number):
        self.logger.debug(f'Setting up game {number}')
        color = random.choice([chess.WHITE, chess.BLACK])
        trainee = ChessEngine(self.trainee_model)
        opponent_model_file = self.get_opponent_model_file()
        return (
            self.cuda_device,
            color,
            trainee,
            self.model,
            opponent_model_file
        )

    def collect_policy_losses(self):
        policy_losses = []
        if self.multi_process:
            with mp.Pool() as p:
                for color, reward, policy_loss in p.imap_unordered(
                    self_play_args, [self.setup_games(n) for n in
                                     range(self.num_games)], mp.cpu_count()):
                    self.self_play_log(color, reward, policy_loss)
                    policy_losses.append(policy_loss)
            return policy_losses
        else:
            for n in range(self.num_games):
                args = self.setup_games(n)
                color, reward, policy_loss = self_play_args(args)
                self.self_play_log(color, reward, policy_loss)
                policy_losses.append(policy_loss)
            return policy_losses

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
        self.logger.info(f'Multi threaded: {self.multi_process}')
        self.logger.info(f'Cuda device: {self.cuda_device}')

        optimizer = optim.SGD(
            self.trainee_model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True
        )
        i = 0
        while i < self.num_iter:
            while True:
                try:
                    policy_losses = self.collect_policy_losses()
                    optimizer.zero_grad()
                    policy_loss = torch.cat(policy_losses).sum()
                    policy_loss /= self.num_games
                    msg = 'Total policy loss for iteration '
                    msg += f'{i}: {policy_loss.data[0]}'
                    if i % self.log_interval == self.log_interval - 1:
                        self.logger.info(msg)
                    else:
                        self.logger.debug(msg)

                    policy_loss.backward()
                    optimizer.step()
                    if i != 0 and i % self.save_interval == 0:
                        self.save(i)
                except PolicyLossIsNan:
                    if self.learning_rate <= 1e-8:
                        self.logger.error(
                            'Policy loss is nan, and learning rate'
                            ' is below 1e-8')
                        raise
                    # policy loss is nan, let's restore the trainee to the
                    # previous state and try again
                    self.logger.info('Policy loss is nan. '
                                     'Retrying with a lower learning rate.')
                    self.logger.info('Restoring to latest model...')
                    self.init_trainee_model_to_latest()
                    self.learning_rate /= 10
                    self.logger.info(
                        f'Reduced learning rate: {self.learning_rate}')
                    i = self.latest_saved_iteration
                    self.logger.info(f'Back to iteration {i}')
                break
            i += 1

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
        self.latest_saved_trainee = filepath
        self.latest_saved_iteration = iteration
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


def self_play_args(args):
    return self_play(*args)


def self_play(
    cuda_device,
    color,
    trainee,
    opponent_model_name,
    opponent_model_file
):
    opponent = models.create(opponent_model_name)
    opponent.load_state_dict(torch.load(opponent_model_file))
    opponent = ChessEngine(opponent, train=False, cuda_device=cuda_device)

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
    if np.isnan(policy_loss.data[0]):
        raise PolicyLossIsNan(log_probs)
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
    parser.add_argument('-c', '--cuda-device', type=int)
    parser.add_argument('-t', '--single-process', action="store_true")
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
    if args.single_process:
        trainer_setting['multi_process'] = not args.single_process
    if args.cuda_device:
        trainer_setting['cuda_device'] = args.cuda_device

    trainer = ReinforceTrainer(**trainer_setting)
    trainer.run()


if __name__ == '__main__':
    run()
