import attr
import os
import chess
import torch
import logging
import random
import glob
import models
import torch.optim as optim
from chess_engine import ChessEngine


@attr.s
class ReinforceTrainer():
    model = attr.ib()
    opponent_pool_path = attr.ib()
    trainee_saved_model = attr.ib()
    learning_rate = attr.ib(default=1e-3)
    num_iter = attr.ib(default=1000)
    num_games = attr.ib(default=128)
    logger = attr.ib(default=logging.getLogger(__name__))

    def __attrs_post_init__(self):
        self.trainee_model = models.create(self.model)
        if self.trainee_saved_model:
            self.trainee_model.load_state_dict(torch.load(self.trainee_saved_model))

    def self_play(self, trainee, opponent, color):
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
        reward = self.get_reward(result, color)
        policy_loss = -torch.cat(log_probs).sum() * (reward - baseline)
        self.self_play_log(color, reward, policy_loss)
        return reward, policy_loss

    def self_play_log(self, color, reward, policy_loss):
        str_color = "white" if color == chess.WHITE else "black"
        self.logger.info(f'Trainee color: {str_color}\tReward: {reward}\t'
            f'Policy loss: {policy_loss.data[0]}')

    def get_reward(self, result, color):
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

    def get_opponent(self):
        opponent_model = models.create(self.model)
        opponent_model_files = glob.glob(os.path.join(
            self.opponent_pool_path, '*.model'))
        opponent_model_file = random.choice(opponent_model_files)
        opponent_model.load_state_dict(torch.load(opponent_model_file))
        return ChessEngine(opponent_model, train=False)

    def game(self):
        trainee_color = random.choice([chess.WHITE, chess.BLACK])
        trainee_engine = ChessEngine(self.trainee_model)
        return self.self_play(trainee_engine, self.get_opponent(), trainee_color)

    def run(self):
        optimizer = optim.Adam(
            self.trainee_model.parameters(), lr=self.learning_rate)
        for i in range(self.num_iter):
            policy_losses = []
            # TODO: parallelize. Possibly using ThreadPoolExecutor
            for g in range(self.num_games):
                _, game_policy_loss = self.game()
                policy_losses.append(game_policy_loss)
            optimizer.zero_grad()
            policy_loss = torch.cat(policy_losses).sum()
            policy_loss /= self.num_games
            self.logger.info(f'Total policy loss for iteration {i}: '
                f'{policy_loss.data[0]}')
            policy_loss.backward()
            optimizer.step()


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

    args = parser.parse_args()

    logger = logging.getLogger('ReinforceTrainer')
    logging_config = {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'level': logging.INFO,
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

    trainer = ReinforceTrainer(**trainer_setting)
    trainer.run()

if __name__ == '__main__':
    run()
