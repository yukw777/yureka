import chess
import torch
import logging


class ReinforceTrainer():

    def self_play(self, trainee, opponent, color):
        logging.info(
            f'Trainee color: {"white" if color == chess.WHITE else "black"}')
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
        return reward, policy_loss

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
