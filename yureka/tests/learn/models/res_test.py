import torch
from yureka.learn import models


def test_res_v0():
    policy, value = models.create_res('ResNet.v0')
    # batch_size * in_channels * 8 * 8
    input = torch.randn(16, 21, 8, 8)
    # batch_size * num_move_planes * 8 * 8
    policy_output = policy(input)
    assert policy_output.shape(16, 73, 8, 8)
