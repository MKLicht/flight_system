from math import pi, cos, sin, sqrt

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agent.utils import to_cpu_tensor, cpu_tensor_to_numpy

xyz_scale = 1.0 / 10000.0
v_scale = 1.0 / 1836.0
pitch_scale = 1.0 / 90.0
yaw_scale = 1.0 / 180.0
roll_scale = 1.0 / 90.0
angle_scale = pi / 180
dtype = np.float32

action_dim = (4, 3)
state_dim = 21


def wrap_action(raw_action: np.ndarray) -> np.ndarray:
    half_value = action_dim[1] // 2
    return (raw_action - half_value) / half_value


# def wrap_angle(yaw, pitch, roll):
#     pitch %= 360
#     yaw %= 360
#     roll %= 360
#     if 90 < pitch <= 270:
#         yaw = (yaw + 180) % 360
#         roll = (roll + 180) % 360
#         pitch = 180 - pitch
#     return yaw, pitch, roll


def dict2obs(state_dict, ind):
    oind = (ind + 1) % 2
    fighters = state_dict['fighters']
    fighter_dict = fighters[ind]
    ofighter_dict = fighters[oind]

    yaw = fighter_dict['yaw'] * angle_scale
    pitch = fighter_dict['pitch'] * angle_scale
    roll = fighter_dict['roll'] * angle_scale
    dx = ofighter_dict['x'] - fighter_dict['x']
    dy = ofighter_dict['y'] - fighter_dict['y']
    dz = ofighter_dict['z'] - fighter_dict['z']

    d = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    oyaw = ofighter_dict['yaw'] * angle_scale
    opitch = ofighter_dict['pitch'] * angle_scale
    oroll = ofighter_dict['roll'] * angle_scale
    obs = np.array(
        [
            fighter_dict['x'] * xyz_scale,
            fighter_dict['y'] * xyz_scale,
            (fighter_dict['z'] - 10000) * 2.0 * xyz_scale,
            fighter_dict['v'] * v_scale,
            sin(pitch),
            sin(yaw),
            cos(yaw),
            sin(roll),
            cos(roll),
            dx * xyz_scale,
            dy * xyz_scale,
            dz * xyz_scale,
            d * xyz_scale,
            ofighter_dict['v'] * v_scale,
            sin(opitch),
            sin(oyaw),
            cos(oyaw),
            sin(oroll),
            cos(oroll),
            cos(fighter_dict['angle1'][0]['angle'] * angle_scale),
            cos(fighter_dict['angle2'][0]['angle'] * angle_scale)
        ],
        dtype=dtype
    )
    return obs


def stochastic_action(policy: nn.Module, obs: np.ndarray):
    with torch.no_grad():
        obs = to_cpu_tensor(obs).unsqueeze(0)
        logits = policy(obs).squeeze(0)
        dist = Categorical(logits=logits)
        action = cpu_tensor_to_numpy(dist.sample())
    return action
