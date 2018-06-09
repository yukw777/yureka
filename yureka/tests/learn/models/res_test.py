import torch
from yureka.learn import models
from yureka.learn.data.move_translator import NUM_MOVE_PLANES


def test_res_v0():
    tower, policy, value = models.create_res('ResNet.v0')
    # (batch_size, in_channels, 8, 8)
    input = torch.randn(16, 21, 8, 8)
    # (batch_size, num_move_planes * 8 * 8)
    policy_output = policy(tower(input))
    assert policy_output.shape == (16, NUM_MOVE_PLANES * 8 * 8)

    # (batch_size, 1)
    value_output = value(tower(input))
    assert value_output.shape == (16, 1)
