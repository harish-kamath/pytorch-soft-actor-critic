import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

def create_arguments(arg_dict):
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    
    custom_args = []
    for k,v in arg_dict.items(): 
        custom_args.append(k)
        custom_args.append(v)
    args = parser.parse_args(custom_args)
    return args

class MBRLEnvWrapper:

    def __init__(self, env, dynamics=None, rewards=None, done_func=None):
        super().__init__()
        self.dynamics = dynamics
        self.rewards = rewards
        self.gym_env = env
        self.done = done_func
        self.current_state = env.reset()
    
    def step(self, action):
        next_state = self.dynamics(self.current_state, action)
        reward = self.rewards(self.current_state, action)
        done = self.done(self.next_state)
        return next_state, reward, done, None
    
    def reset(self):
        self.current_state = env.reset()
    
    def close(self):
        self.reset()

class SoftActorCritic:
    
    def __init__(self, env, wandb_writer, kwargs=None):
        super().__init__()
        args = create_arguments(kwargs)
        self.env = env
        self.agent = SAC(env.gym_env.observation_space.shape[0], env.gym_env.action_space, args)
        self.wandb_writer = wandb_writer
        self.memory = ReplayMemory(args.replay_size, args.seed)
        self.args = args

    def train(self, num_steps=None):

        agent = self.agent
        env = self.env
        args = self.args
        writer = self.wandb_writer
        if not num_steps: num_steps = self.args.num_steps

        total_numsteps = 0
        updates = 0

        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            state = env.gym_env.reset()

            while not done:
                if args.start_steps > total_numsteps:
                    action = env.gym_env.action_space.sample()  # Sample random action
                else:
                    action = agent.select_action(state)  # Sample action from policy

                if len(memory) > args.batch_size:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                        writer.log({
                            "critic_1": critic_1_loss,
                            "critic_2": critic_2_loss,
                            "policy": policy_loss,
                            "entropy_loss": ent_loss,
                            "alpha (entropy temperature)": alpha 
                        }, step=updates)
                        updates += 1

                next_state, reward, done, _ = env.step(action) # Step
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == env.gym_env._max_episode_steps else float(not done)

                memory.push(state, action, reward, next_state, mask) # Append transition to memory

                state = next_state

            if total_numsteps > args.num_steps:
                break
            writer.log({"train reward": episode_reward}, step=i_episode)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

            if i_episode % 10 == 0 and args.eval is True:
                avg_reward = 0.
                episodes = 10
                for _  in range(episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = agent.select_action(state, evaluate=True)

                        next_state, reward, done, _ = env.step(action)
                        episode_reward += reward


                        state = next_state
                    avg_reward += episode_reward
                avg_reward /= episodes

                writer.log({"test average reward": avg_reward}, step=i_episode)

                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
                print("----------------------------------------")

        env.close()

