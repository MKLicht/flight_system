import json
import random

from env.fighter import Fighter
from env.result import get_result
from env.server_thread import ServerThread


class AirCombatEnv:
    def __init__(self, config='test', server_thread: ServerThread = None):
        with open('config/{}.json'.format(config), "r") as f:
            self.config = json.load(f)
        self.server_thread = server_thread

        if self.config['render'] != 0:
            self.server_thread.setDaemon(True)
            self.server_thread.start()

        self.max_step = self.config['max_step']
        self.random_pos = self.config['random_pos'] != 0
        self.red_num = self.config['red_num']
        self.fighter_num = len(self.config['fighters'])
        self.dt = self.config['dt']
        self.delay = self.config['delay']
        self.reward = self.config['reward']
        self.fighters = [Fighter() for i in range(self.fighter_num)]
        self.state_dict = None

    def reset(self):
        self.state_dict = {
            "current_step": 0,
            'dt': self.dt,
            "delay": self.delay,
            "red_num": self.red_num,
            "fighters": []
        }

        for i in range(self.fighter_num):
            config_fighter_dict = self.config['fighters'][i]
            fighter = self.fighters[i]
            if self.random_pos:
                x = random.uniform(-9000, 9000)
                y = random.uniform(-9000, 9000)
                yaw = random.uniform(-180, 180)
            else:
                x = config_fighter_dict['x']
                y = config_fighter_dict['y']
                yaw = config_fighter_dict['yaw']
            z = config_fighter_dict['z']
            v = config_fighter_dict['v']
            fighter_dict = {
                'is_red': i < self.red_num,
                'dead': False,
                'out': False,
                'lock': 0,
                'locked': 0,
                'angle1': [],
                'angle2': [],
                'reward': 0
            }
            fighter.reset([y, x, z, yaw, 0, 0, 0, v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            fighter.state(fighter_dict)
            self.state_dict['fighters'].append(fighter_dict)

        get_result(self.state_dict)
        return self.state_dict

    def step(self, actions):
        self.state_dict['current_step'] += 1
        for i in range(self.fighter_num):
            fighter_dict = self.state_dict['fighters'][i]
            fighter = self.fighters[i]
            if fighter_dict['dead']:
                continue
            action = actions[i]
            for j in range(self.delay):
                fighter.step(self.dt, action)
            fighter.state(fighter_dict)

            fighter_dict['lock'] = 0
            fighter_dict['locked'] = 0
            fighter_dict['angle1'].clear()
            fighter_dict['angle2'].clear()

        get_result(self.state_dict)
        reward = [0.0] * self.fighter_num
        red_done, blue_done = True, True
        for i, fighter_dict in enumerate(self.state_dict['fighters']):
            if fighter_dict['dead']:
                continue
            reward[i] = fighter_dict['lock'] * self.reward['lock'] + fighter_dict['locked'] * self.reward[
                'locked'] + fighter_dict['out'] * self.reward['out']
            fighter_dict['reward'] += reward[i]
            if i < self.red_num:
                red_done = False
            else:
                blue_done = False

        done = red_done or blue_done or self.state_dict['current_step'] >= self.max_step
        info = 0
        if done:
            if red_done:
                info -= 1
            if blue_done:
                info += 1

        return self.state_dict, reward, done, info

    def render(self):
        if self.config['render'] != 0:
            self.server_thread.send(self.state_dict)
