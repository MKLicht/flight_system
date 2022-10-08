import os

import numpy as np
import torch

from agent.agent import stochastic_action
from agent.ppo.network import Policy
from agent.utils import to_cpu_tensor, cpu_tensor_to_numpy

dtype = np.float32

class Trainer:
    def __init__(
            self,
            state_dim,
            action_dim,
            path,
            gamma=0.99,
            lr_actor=3e-4,
            load_model=True
    ):
        self.lr_actor = lr_actor
        self.gamma = gamma

        self.policy = Policy(state_dim, action_dim).to('cpu')
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.counter = 0
        self.model_path = path + '/reinforce.pt'
        if not os.path.exists(path):
            os.makedirs(path)
        if load_model:
            if os.path.exists(self.model_path):
                model = torch.load(self.model_path)
                self.counter = model['counter']
                self.policy.load_state_dict(model['policy'])

    def get_action(self, obs):
        action = stochastic_action(self.policy, obs)
        return action

    def compute_log_probs(self, observation, action):
        observation=self.to_tensor(observation).view(-1,self.obs_dim)
        action=self.to_tensor(action)
        return self.policy.log_prob(observation,action)

    def learn(self, rewards, states, dones, actions):
        rewards = np.asarray(rewards, dtype=dtype)
        dones = np.asarray(dones, dtype=dtype)
        batch_state = to_cpu_tensor(np.asarray(states, dtype=dtype))
        batch_action = to_cpu_tensor(np.asarray(actions))

        # TODO:使用rewards和计算return
        returns = np.zeros_like(rewards, dtype = np.float32)
        r = 0
        size = len(rewards)
        returns[-1] = rewards[-1]
        for i in reversed(range(0, len(rewards)-1)):
            returns[i] = rewards[i] + (1-dones[i]) * self.gamma * returns[i+1]
        mean = np.mean(returns)
        std = np.std(returns) + 1e-6
        returns=(returns - mean) / (std + 1e-6)
        batch_return = to_cpu_tensor(returns)

        # TODO:计算策略loss
        logits = self.policy.forward(batch_state)
        log_pi = torch.distributions.Categorical(logits=logits).log_prob(batch_action).sum(1)
        actor_loss = -(log_pi * batch_return).mean()

        # update actor
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()
        self.counter += 1

        return actor_loss.item()

    def save(self):
        model = {
            'counter': self.counter,
            'policy': self.policy.state_dict()
        }
        torch.save(model, self.model_path)
