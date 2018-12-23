import torch
import torch.nn as nn
import os

from yureka.learn import models
from yureka.engine import constants


class Network(nn.Module):
    def __init__(self, tower, policy, value):
        super(Network, self).__init__()
        self.tower = tower
        self.policy = policy
        self.value = value

    def forward(self, x):
        x = self.tower(x)
        return self.policy(x), self.value(x)


tower, policy, value = models.create(constants.DEFAULT_RESNET)
tower.load_state_dict(
    torch.load(os.path.expanduser(constants.DEFAULT_RESNET_TOWER_FILE)))
policy.load_state_dict(
    torch.load(os.path.expanduser(constants.DEFAULT_RESNET_POLICY_FILE)))
value.load_state_dict(
    torch.load(os.path.expanduser(constants.DEFAULT_RESNET_VALUE_FILE)))

network = Network(tower, policy, value).eval()
input = torch.randn(16, 21, 8, 8)
traced_script_module = torch.jit.trace(network, input)
traced_script_module.save("model.pt")
