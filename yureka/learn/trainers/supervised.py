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

from .. import models
from ..data.chess_dataset import LMDBChessDataset


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
    logger = attr.ib(default=logging.getLogger(__name__))
    log_interval = attr.ib(default=2000)
    batch_size = attr.ib(default=16)
    num_epochs = attr.ib(default=100)
    cuda = attr.ib(default=True)
    parallel = attr.ib(default=False)
    learning_rate = attr.ib(default=1e-4)
    value = attr.ib(default=False)

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
            self.model.to(self.device)
        self.train_data, self.test_data = self.split_train_test(self.data)
        self.logger.info(f'Train data len: {len(self.train_data)}')
        self.logger.info(f'Test data len: {len(self.test_data)}')

        self.lr_reduced = False
        self.original_learning_rate = self.learning_rate
        self.criterion = nn.MSELoss() if self.value else nn.CrossEntropyLoss()

    def print_summary(self):
        self.logger.info(f'Data: {self.data}')
        self.logger.info(f'Test ratio: {self.test_ratio}')
        self.logger.info(f'Log interval: {self.log_interval}')
        self.logger.info(f'Batch size: {self.batch_size}')
        self.logger.info(f'Num epochs: {self.num_epochs}')
        self.logger.info(f'Use cuda: {self.cuda}')
        self.logger.info(f'Learning rate: {self.learning_rate}')

    def split_train_test(self, data_files):
        test = []
        train = []
        for f in data_files:
            temp = LMDBChessDataset(f)
            test_len = round(len(temp) * self.test_ratio)
            del temp
            test.append(LMDBChessDataset(f, limit=test_len))
            train.append(LMDBChessDataset(f, offset=test_len))

        return data.DataLoader(
            data.ConcatDataset(train),
            batch_size=self.batch_size,
            num_workers=4
        ), data.DataLoader(
            data.ConcatDataset(test),
            batch_size=self.batch_size,
        )

    def get_variables_from_inputs(self, row):
        # get the inputs
        inputs, move, value = row

        if self.value:
            labels = value
        else:
            labels = move

        return inputs.to(self.device), labels.to(self.device)

    def predict(self, inputs):
        outputs = self.model(inputs)
        if not self.value:
            return outputs.view(outputs.shape[0], -1)
        return outputs

    def run(self):
        for epoch in range(self.num_epochs):
            self.logger.info(f'Epoch {epoch}')
            if self.lr_reduced:
                self.logger.info('Restoring reduced learning rate')
                self.learning_rate = self.original_learning_rate
                self.lr_reduced = False
            while True:
                try:
                    self.train(epoch)
                except LossIsNan:
                    if self.learning_rate <= 1e-6:
                        self.logger.error(
                            'Loss is nan, and learning rate is below 1e-6')
                        raise
                    # loss is nan, try again with a reduced learning rate
                    self.logger.info('Loss is nan. Reducing learning rate')
                    self.logger.info(
                        f'Current learning rate: {self.learning_rate}')
                    self.learning_rate /= 10
                    self.logger.info(
                        f'Reduced learning rate: {self.learning_rate}')
                    self.lr_reduced = True
                break
            self.save(epoch)
            self.test(epoch)

    def save(self, epoch):
        filename = self.model.name
        filename += f"_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}"
        filename += f"_{epoch}.model"
        filepath = os.path.join(
            os.getcwd(),
            'saved_models',
            filename
        )
        self.logger.info(f'Saving: {filepath}')
        torch.save(self.model.state_dict(), filepath)
        self.logger.info('Done saving')

    def test(self, epoch):
        self.logger.info('Testing...')
        self.model.eval()

        losses = []
        if not self.value:
            predictions = np.array([])
            answers = np.array([])
        for i, row in enumerate(self.test_data):
            inputs, labels = self.get_variables_from_inputs(row)

            # loss
            outputs = self.predict(inputs)
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())

            if not self.value:
                _, prediction = outputs.max(1)
                predictions = np.append(predictions, prediction.item())
                answers = np.append(answers, labels.item())

        avg_loss = np.average(losses)
        if not self.value:
            precision = metrics.precision_score(
                answers,
                predictions,
                average='micro'
            )
            recall = metrics.recall_score(
                answers, predictions, average='micro')
            f1_score = metrics.f1_score(answers, predictions, average='micro')
        self.logger.info(f'Avg. loss at epoch {epoch}: {avg_loss}')
        if not self.value:
            self.logger.info(f'Precision at epoch {epoch}: {precision}')
            self.logger.info(f'Recall at epoch {epoch}: {recall}')
            self.logger.info(f'F1 score at epoch {epoch}: {f1_score}')
        self.logger.info('Testing finished')

    def train(self, epoch):
        self.logger.info('Training...')
        self.model.train()

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True
        )

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
    parser.add_argument('-d', '--data', action='append', required=True)
    parser.add_argument('-i', '--log-interval', type=int)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-e', '--num-epochs', type=int)
    parser.add_argument('-p', '--parallel', action='store_true')
    parser.add_argument('-l', '--log-file')
    parser.add_argument('-s', '--saved-model')
    parser.add_argument('-r', '--learning-rate', type=float)
    parser.add_argument('-v', '--value', action='store_true')

    args = parser.parse_args()

    logger = logging.getLogger('SupervisedTrainer')
    logging_config = {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'level': logging.INFO,
    }
    if args.log_file:
        logging_config['filename'] = args.log_file
    logging.basicConfig(**logging_config)

    model = models.create(args.model)
    if args.saved_model:
        logger.info(f'Loading saved model: {args.saved_model}')
        model.load_state_dict(torch.load(args.saved_model))

    trainer_setting = {
        'model': model,
        'data': args.data,
        'test_ratio': args.test_ratio,
        'logger': logger,
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
    if args.value:
        trainer_setting['value'] = args.value
    trainer = SupervisedTrainer(**trainer_setting)
    trainer.run()


if __name__ == '__main__':
    run()
