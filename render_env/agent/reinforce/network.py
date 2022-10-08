import torch.nn as nn

from agent.utils import MLPNetwork


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim: tuple, hidden_size=256):
        super(Policy, self).__init__()
        self.action_num, self.action_seg = action_dim
        self.network = MLPNetwork(state_dim, self.action_num * self.action_seg, hidden_size)

    def forward(self, obs):
        logits = self.network(obs).view(-1, self.action_num, self.action_seg)
        return logits
