import logging
import attr
import argparse
import datetime
import os
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
import sklearn.metrics as metrics
from torch.autograd import Variable
from chess_dataset import ChessDataset


@attr.s
class LossIsNan(Exception):
    inputs = attr.ib()
    outputs = attr.ib()
    labels = attr.ib()
    loss = attr.ib()


@attr.s
class SupervisedTrainer():
    model = attr.ib()
    train_data = attr.ib()
    test_data = attr.ib()
    logger = attr.ib(default=logging.getLogger(__name__))
    log_interval = attr.ib(default=2000)
    batch_size = attr.ib(default=16)
    num_epochs = attr.ib(default=100)
    cuda = attr.ib(default=True)
    learning_rate = 1e-4

    def __attrs_post_init__(self):
        self.cuda = self.cuda and torch.cuda.is_available()
        # summary
        self.print_summary()

        if self.cuda:
            self.logger.info('Using CUDA')
            self.model.cuda()
        self.train_data = self.get_data_loader(self.train_data)
        self.test_data = self.get_data_loader(self.test_data)

    def print_summary(self):
        self.logger.info(f'Train data: {self.train_data}')
        self.logger.info(f'Test data: {self.test_data}')
        self.logger.info(f'Log interval: {self.log_interval}')
        self.logger.info(f'Batch size: {self.batch_size}')
        self.logger.info(f'Num epochs: {self.num_epochs}')
        self.logger.info(f'Use cuda: {self.cuda}')
        self.logger.info(f'Learning rate: {self.learning_rate}')

    def get_data_loader(self, data_file):
        dataset = ChessDataset(data_file)
        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def get_variables_from_inputs(self, row):
        # get the inputs
        inputs, labels = row

        volatile = not self.model.training

        # wrap them in Variable
        if self.cuda:
            inputs = Variable(inputs.cuda(), volatile=volatile)
            labels = Variable(labels.cuda(), volatile=volatile)
        else:
            inputs = Variable(inputs, volatile=volatile)
            labels = Variable(labels, volatile=volatile)

        return inputs, labels

    def predict(self, inputs):
        outputs = self.model(inputs)
        return outputs.view(outputs.shape[0], -1)

    def run(self):
        for epoch in range(self.num_epochs):
            self.logger.info(f'Epoch {epoch}')
            self.train(epoch)
            self.save(epoch)
            self.test(epoch)

    def save(self, epoch):
        filename = self.model.__class__.__name__
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
        predictions = np.array([])
        answers = np.array([])
        for i, row in enumerate(self.test_data):
            inputs, labels = self.get_variables_from_inputs(row)

            # loss
            outputs = self.predict(inputs)
            loss = F.cross_entropy(outputs, labels)
            losses.append(loss.data)

            _, prediction = outputs.max(1)
            predictions = np.append(predictions, prediction.data)
            answers = np.append(answers, labels.data)

        avg_loss = np.average(losses)
        precision = metrics.precision_score(
            answers,
            predictions,
            average='micro'
        )
        recall = metrics.recall_score(answers, predictions, average='micro')
        f1_score = metrics.f1_score(answers, predictions, average='micro')
        self.logger.info(f'Avg. loss at epoch {epoch}: {avg_loss}')
        self.logger.info(f'Precision at epoch {epoch}: {precision}')
        self.logger.info(f'Recall at epoch {epoch}: {recall}')
        self.logger.info(f'F1 score at epoch {epoch}: {f1_score}')
        self.logger.info('Testing finished')

    def train(self, epoch):
        self.logger.info('Training...')
        self.model.train()

        criterion = nn.CrossEntropyLoss()
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
            outputs = self.model(inputs)
            outputs = outputs.view(outputs.shape[0], -1)

            loss = criterion(outputs, labels)
            if np.isnan(loss.data[0]):
                # oops, loss is nan, probably means gradient exploded
                # let's try again with a lower learning rate
                raise LossIsNan(inputs, outputs, labels, loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % self.log_interval == self.log_interval - 1:
                avg_loss = running_loss / self.log_interval
                self.logger.info('[%d, %5d] loss: %.3f' %
                                 (epoch, i, avg_loss))
                running_loss = 0.0

        self.logger.info('Training finished')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('train_data')
    parser.add_argument('test_data')
    parser.add_argument('-i', '--log-interval', type=int)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-e', '--num-epochs', type=int)
    parser.add_argument('-c', '--cuda', type=bool)
    parser.add_argument('-l', '--log-file')
    parser.add_argument('-s', '--saved-model')
    parser.add_argument('-r', '--learning-rate', type=float)

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
        'train_data': args.train_data,
        'test_data': args.test_data,
        'logger': logger,
    }
    if args.log_interval:
        trainer_setting['log_interval'] = args.log_interval
    if args.batch_size:
        trainer_setting['batch_size'] = args.batch_size
    if args.num_epochs:
        trainer_setting['num_epochs'] = args.num_epochs
    if args.cuda:
        trainer_setting['cuda'] = args.cuda
    if args.learning_rate:
        trainer_setting['learning_rate'] = args.learning_rate
    trainer = SupervisedTrainer(**trainer_setting)
    trainer.run()


if __name__ == '__main__':
    run()
