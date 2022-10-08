import os
import numpy as np
import torch
import torch.nn as nn

from agent.agent import stochastic_action, wrap_action, dict2obs
from agent.reinforce.network import Policy


def get_action(policy: nn.Module, obs: np.ndarray):
    raw_action = stochastic_action(policy, obs)
    action = wrap_action(raw_action).tolist()
    return action


def load_policy(state_dim, action_dim, path='') -> nn.Module:
    policy = Policy(state_dim, action_dim).to('cpu')
    model_path = path + '/reinforce.pt'
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=torch.device('cpu'))
        policy.load_state_dict(model['policy'])
    return policy


def evaluate(env, policy_red: nn.Module, policy_blue: nn.Module):
    state_dict = env.reset()
    total_reward_red = 0
    total_reward_blue = 0
    while True:
        obs_red = dict2obs(state_dict, 0)
        obs_blue = dict2obs(state_dict, 1)
        action_red = get_action(policy_red, obs_red)
        action_blue = get_action(policy_blue, obs_blue)
        state_dict, reward, done, info = env.step([action_red, action_blue])
        total_reward_red += reward[0]
        total_reward_blue += reward[1]
        if done:
            break
    return [total_reward_red, total_reward_blue], info

