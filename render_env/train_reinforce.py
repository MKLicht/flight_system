import numpy as np
from mpi4py import MPI
from tensorboardX import SummaryWriter

from agent import dict2obs, wrap_action, state_dim, action_dim
from agent.reinforce import Trainer
from env import AirCombatEnv

actions = [[0, 0, 0, 0], [0, 0, 0, 0]]
env = AirCombatEnv(config='train')

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
path = 'log/' + str(rank)
writer = SummaryWriter(path)
trainer = Trainer(
    state_dim=state_dim,
    action_dim=action_dim,
    path=path,
    load_model=True
)

red_rewards = []
red_states = []
red_dones = []
red_actions = []

blue_rewards = []
blue_states = []
blue_dones = []
blue_actions = []

track_reward = []
track_actor_loss = []
episode = 0
iteration = trainer.counter
while True:
    state_dict = env.reset()

    total_reward_red = 0
    total_reward_blue = 0
    actor_loss_list = []
    while True:
        obs_red = dict2obs(state_dict, 0)
        obs_blue = dict2obs(state_dict, 1)
        act_red = trainer.get_action(obs_red)
        act_blue = trainer.get_action(obs_blue)
        actions[0][0], actions[0][1], actions[0][2], actions[0][3] = wrap_action(act_red)
        actions[1][0], actions[1][1], actions[1][2], actions[1][3] = wrap_action(act_blue)
        red_states.append(obs_red)
        red_actions.append(act_red)
        blue_states.append(obs_blue)
        blue_actions.append(act_blue)
        state_dict, reward, done, _ = env.step(actions)
        reward_red, reward_blue = reward

        red_rewards.append(reward_red)
        red_dones.append(done)
        blue_rewards.append(reward_blue)
        blue_dones.append(done)

        total_reward_red += reward_red
        total_reward_blue += reward_blue

        if done:
            rewards = red_rewards + blue_rewards
            states = red_states + blue_states
            dones = red_dones + blue_dones
            raw_actions = red_actions + blue_actions
            red_rewards.clear()
            red_states.clear()
            red_dones.clear()
            red_actions.clear()

            blue_rewards.clear()
            blue_states.clear()
            blue_dones.clear()
            blue_actions.clear()

            actor_loss = trainer.learn(rewards, states, dones, raw_actions)
            track_reward.append(total_reward_red)
            track_reward.append(total_reward_blue)
            track_actor_loss.append(actor_loss)
            iteration += 1
            if iteration % 50 == 0:
                writer.add_scalar(tag='reward', scalar_value=np.mean(track_reward), global_step=iteration)
                writer.add_scalar(tag='actor loss', scalar_value=np.mean(track_actor_loss), global_step=iteration)
                track_reward.clear()
                track_actor_loss.clear()
                trainer.save()
                print('iteration: {}, save.'.format(iteration))
            break

    episode += 1
    print('episode: {}, red: {}, blue: {}.'.format(episode, total_reward_red, total_reward_blue))
