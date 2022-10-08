import time

from agent import dict2obs, action_dim, state_dim
from agent.reinforce import load_policy, get_action
from env import AirCombatEnv as AirCombatEnv
from env import InteractServerThread

actions = [[-1, 0, 0, 0], [-1, 0, 0, 0]]
server_thread = InteractServerThread(actions=actions, mask=[True, False], port=9020)
env = AirCombatEnv(server_thread=server_thread)

episode_num = 10

policy_red = load_policy(
    state_dim=state_dim,
    action_dim=action_dim,
    path='log/0'
)

policy_blue = load_policy(
    state_dim=state_dim,
    action_dim=action_dim,
    path='log/2'
)

for i in range(episode_num):
    state_dict = env.reset()
    env.render()
    total_reward_red = 0
    total_reward_blue = 0
    while True:
        obs_red = dict2obs(state_dict, 0)
        obs_blue = dict2obs(state_dict, 1)
        actions[0][0], actions[0][1], actions[0][2], actions[0][3] = get_action(policy_red, obs_red)
        actions[1][0], actions[1][1], actions[1][2], actions[1][3] = get_action(policy_blue, obs_blue)
        state_dict, reward, done, info = env.step(actions)
        env.render()
        time.sleep(0.05)
        total_reward_red += reward[0]
        total_reward_blue += reward[1]
        if done:
            break
    print('episode: {}, red: {}, blue: {}.'.format(i, total_reward_red, total_reward_blue))
