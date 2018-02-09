import models
import torch
from torch.autograd import Variable


def test_policy_v0():
    m = models.create('Policy.v0')
    # batch_size * in_channels * 8 * 8
    input = Variable(torch.randn(16, 23, 8, 8))
    # batch_size * num_move_planes * 8 * 8
    output = m(input)
    assert output.shape == (16, 73, 8, 8)


def test_policy_v1():
    m = models.create('Policy.v1')
    assert m.batch_norm
    # batch_size * in_channels * 8 * 8
    input = Variable(torch.randn(16, 23, 8, 8))
    # batch_size * num_move_planes * 8 * 8
    output = m(input)
    assert output.shape == (16, 73, 8, 8)


def test_value_v0():
    m = models.create('Value.v0')
    # batch_size * in_channels * 8 * 8
    input = Variable(torch.randn(16, 23, 8, 8))
    # batch_size * 1
    output = m(input)
    assert output.shape == (16, 1)


def test_value_v1():
    m = models.create('Value.v1')
    assert m.batch_norm
    # batch_size * in_channels * 8 * 8
    input = Variable(torch.randn(16, 23, 8, 8))
    # batch_size * 1
    output = m(input)
    assert output.shape == (16, 1)


def test_rollout_v0():
    m = models.create('Rollout.v0')
    # batch_size * in_channels * 8 * 8
    input = Variable(torch.randn(16, 23, 8, 8))
    # batch_size * num_move_planes * 8 * 8
    output = m(input)
    assert output.shape == (16, 73, 8, 8)


def test_rollout_v1():
    m = models.create('Rollout.v1')
    # batch_size * in_channels * 8 * 8
    input = Variable(torch.randn(16, 23, 8, 8))
    # batch_size * num_move_planes * 8 * 8
    output = m(input)
    assert output.shape == (16, 73, 8, 8)
