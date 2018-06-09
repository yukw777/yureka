import logging
import attr
import argparse
import datetime
import os
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics

from .loss import MSECrossEntropyLoss
from .. import models
from ..data.chess_dataset import (
    LMDBChessDataset,
    InterleavenDataset,
    ChessDataset,
)


@attr.s
class LossIsNan(Exception):
    inputs = attr.ib()
    outputs = attr.ib()
    labels = attr.ib()
    loss = attr.ib()


@attr.s
class SupervisedTrainer():
    model = attr.ib()
    data = attr.ib()
    test_ratio = attr.ib()
    model_path = attr.ib()
    format = attr.ib(default='csv')
    network = attr.ib(default='res')
    data_limit = attr.ib(default=None)
    logger = attr.ib(default=logging.getLogger(__name__))
    log_interval = attr.ib(default=2000)
    batch_size = attr.ib(default=16)
    num_epochs = attr.ib(default=100)
    cuda = attr.ib(default=True)
    parallel = attr.ib(default=False)
    learning_rate = attr.ib(default=1e-2)

    def __attrs_post_init__(self):
        self.cuda = self.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        # summary
        self.print_summary()

        if self.parallel and torch.cuda.device_count() > 1:
            device_count = torch.cuda.device_count()
            self.logger.info(f'Using {device_count} GPUs')
            self.model = nn.DataParallel(self.model)
        else:
            if self.network == 'res':
                tower, policy, value = self.model
                tower.to(self.device)
                policy.to(self.device)
                value.to(self.device)
            else:
                self.model.to(self.device)
        self.train_data, self.test_data = self.split_train_test(
            self.data, self.data_limit)
        self.logger.info(f'Train data len: {len(self.train_data)}')
        self.logger.info(f'Test data len: {len(self.test_data)}')

        if self.network == 'value':
            self.criterion = nn.MSELoss()
        elif self.network == 'policy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.network == 'res':
            self.criterion = MSECrossEntropyLoss()

    def print_summary(self):
        self.logger.info(f'Data: {self.data}')
        self.logger.info(f'Test ratio: {self.test_ratio}')
        self.logger.info(f'Log interval: {self.log_interval}')
        self.logger.info(f'Batch size: {self.batch_size}')
        self.logger.info(f'Num epochs: {self.num_epochs}')
        self.logger.info(f'Use cuda: {self.cuda}')
        self.logger.info(f'Learning rate: {self.learning_rate}')

    def split_train_test(self, data_files, limit=None):
        test = []
        train = []
        for f in data_files:
            if self.format == 'csv':
                temp = ChessDataset(f)
            else:
                temp = LMDBChessDataset(f)
            if limit:
                test_len = round(limit * self.test_ratio)
            else:
                test_len = round(len(temp) * self.test_ratio)
            del temp
            if self.format == 'csv':
                test.append(ChessDataset(f, limit=test_len))
                train.append(ChessDataset(f, limit=limit, offset=test_len))
            elif self.format == 'lmdb':
                test.append(LMDBChessDataset(f, limit=test_len))
                train.append(LMDBChessDataset(f, limit=limit, offset=test_len))

        if len(train) == 1:
            train_dataset = train[0]
        elif len(train) == 2:
            train_dataset = InterleavenDataset(train)
        else:
            train_dataset = data.ConcatDataset(train)

        return data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=(self.format == 'csv')
        ), data.DataLoader(
            data.ConcatDataset(test),
            batch_size=self.batch_size,
        )

    def get_variables_from_inputs(self, row):
        # get the inputs
        inputs, move, value = row
        inputs = inputs.to(self.device)

        if self.network == 'value':
            labels = value.to(self.device)
        elif self.network == 'policy':
            labels = move.to(self.device)
        elif self.network == 'res':
            labels = (move.to(self.device), value.to(self.device))

        return inputs, labels

    def predict(self, inputs):
        if self.network == 'res':
            tower, policy, value = self.model
            tower_output = tower(inputs)
            return (policy(tower_output), value(tower_output))
        else:
            outputs = self.model(inputs)
            if self.network == 'policy':
                return outputs.view(outputs.shape[0], -1)
            if self.network == 'value':
                return outputs

    def run(self):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=1
        )
        for epoch in range(self.num_epochs):
            self.logger.info(f'Epoch {epoch}')
            self.train(epoch, optimizer)
            self.save(epoch)
            loss = self.test(epoch)
            scheduler.step(loss)

    def save(self, epoch):
        filename = self.model.name
        filename += f"_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}"
        filename += f"_{epoch}.model"
        filepath = os.path.join(self.model_path, filename)
        self.logger.info(f'Saving: {filepath}')
        torch.save(self.model.state_dict(), filepath)
        self.logger.info('Done saving')

    def test(self, epoch):
        self.logger.info('Testing...')
        self.model.eval()

        losses = []
        mse_losses = []
        cross_entropy_losses = []
        policy_or_res = self.network in ('res', 'policy')
        if policy_or_res:
            predictions = np.array([])
            answers = np.array([])
        for i, row in enumerate(self.test_data):
            inputs, labels = self.get_variables_from_inputs(row)

            # loss
            outputs = self.predict(inputs)
            loss = self.criterion(outputs, labels)
            if self.network == 'res':
                total_loss, mse_loss, cross_entropy_loss = loss
                losses.append(total_loss.item())
                mse_losses.append(mse_loss.item())
                cross_entropy_losses.append(cross_entropy_loss.item())
            else:
                losses.append(loss.item())

            if policy_or_res:
                if self.network == 'res':
                    move_outputs, _ = outputs
                    _, prediction = move_outputs.max(1)
                else:
                    _, prediction = outputs.max(1)
                predictions = np.append(predictions, prediction)
                if self.network == 'res':
                    move_labels, _ = answers
                    answers = np.append(answers, move_labels)
                else:
                    answers = np.append(answers, labels)

        avg_loss = np.average(losses)
        if self.network == 'res':
            avg_mse_loss = np.average(mse_losses)
            avg_cross_entropy_loss = np.average(cross_entropy_losses)
        if policy_or_res:
            precision = metrics.precision_score(
                answers,
                predictions,
                average='micro'
            )
            recall = metrics.recall_score(
                answers, predictions, average='micro')
            f1_score = metrics.f1_score(answers, predictions, average='micro')
        self.logger.info(f'Avg. loss at epoch {epoch}: {avg_loss}')
        if self.network == 'res':
            self.logger.info(f'Avg. mse loss at epoch {epoch}: {avg_mse_loss}')
            self.logger.info(
                f'Avg. cross entropy loss at epoch {epoch}: '
                f'{avg_cross_entropy_loss}')
        if policy_or_res:
            self.logger.info(f'Precision at epoch {epoch}: {precision}')
            self.logger.info(f'Recall at epoch {epoch}: {recall}')
            self.logger.info(f'F1 score at epoch {epoch}: {f1_score}')
        self.logger.info('Testing finished')
        return avg_loss

    def train(self, epoch, optimizer):
        self.logger.info('Training...')
        self.model.train()

        running_loss = 0.0
        for i, row in enumerate(self.train_data):
            # get the inputs and labels
            inputs, labels = self.get_variables_from_inputs(row)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.predict(inputs)

            loss = self.criterion(outputs, labels)
            if np.isnan(loss.item()):
                # oops, loss is nan, probably means gradient exploded
                # let's try again with a lower learning rate
                raise LossIsNan(inputs, outputs, labels, loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % self.log_interval == self.log_interval - 1:
                avg_loss = running_loss / self.log_interval
                self.logger.info('[%d, %5d] loss: %.3f' %
                                 (epoch, i, avg_loss))
                running_loss = 0.0

        self.logger.info('Training finished')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('test_ratio', type=float)
    parser.add_argument('model_path')
    parser.add_argument('--data-limit', type=int)
    parser.add_argument('-d', '--data', action='append', required=True)
    parser.add_argument('-f', '--format', default='csv')
    parser.add_argument('-i', '--log-interval', type=int)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-e', '--num-epochs', type=int)
    parser.add_argument('-p', '--parallel', action='store_true')
    parser.add_argument('-l', '--log-file')
    parser.add_argument('-s', '--saved-model')
    parser.add_argument('--saved-tower-model')
    parser.add_argument('--saved-policy-model')
    parser.add_argument('--saved-value-model')
    parser.add_argument('-r', '--learning-rate', type=float)
    parser.add_argument('-n', '--network', default='res')

    args = parser.parse_args()

    logger = logging.getLogger('SupervisedTrainer')
    logging_config = {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'level': logging.INFO,
    }
    if args.log_file:
        logging_config['filename'] = args.log_file
    logging.basicConfig(**logging_config)

    if args.network == 'res':
        tower, policy, value = models.create_res(args.model)
        if args.saved_tower_model:
            logger.info(f'Loading saved tower model: {args.saved_tower_model}')
            tower.load_state_dict(torch.load(args.saved_tower_model))
        if args.saved_policy_model:
            logger.info(
                f'Loading saved policy head model: {args.saved_policy_model}')
            policy.load_state_dict(torch.load(args.saved_policy_model))
        if args.saved_value_model:
            logger.info(
                f'Loading saved value head model: {args.saved_value_model}')
            value.load_state_dict(torch.load(args.saved_value_model))
        model = (tower, policy, value)
    else:
        model = models.create(args.model)
        if args.saved_model:
            logger.info(f'Loading saved model: {args.saved_model}')
            model.load_state_dict(torch.load(args.saved_model))

    trainer_setting = {
        'model': model,
        'data': args.data,
        'format': args.format,
        'test_ratio': args.test_ratio,
        'model_path': args.model_path,
        'logger': logger,
        'network': args.network,
    }
    if args.log_interval:
        trainer_setting['log_interval'] = args.log_interval
    if args.batch_size:
        trainer_setting['batch_size'] = args.batch_size
    if args.num_epochs:
        trainer_setting['num_epochs'] = args.num_epochs
    if args.parallel:
        trainer_setting['parallel'] = args.parallel
    if args.learning_rate:
        trainer_setting['learning_rate'] = args.learning_rate
    if args.data_limit:
        trainer_setting['data_limit'] = args.data_limit
    trainer = SupervisedTrainer(**trainer_setting)
    trainer.run()


if __name__ == '__main__':
    run()
