from unityagents import UnityEnvironment
from agents import PPO_Agent
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parameters for Continuous Control (PPO Agent)')
parser.add_argument('-n', '--n_episode', type=int,help='Input the number of episodes you want')
parser.add_argument('-p', '--path', type=str,help='the path of environment file' )
parser.add_argument('-l1', '--learning_time', type=int,help='How many times to learn in each learning behavior' )
parser.add_argument('-lr1', '--learning_rate1', type=float,help='learning rate for actor' )
parser.add_argument('-lr2', '--learning_rate2', type=float,help='learning rate for critic' )
parser.add_argument('-b', '--beta', type=float,help='Beta for entropy loss' )
parser.add_argument('-s', '--std', type=float,help='Standard deviation for initial noise generator' )
parser.add_argument('-e', '--eps', type=float,help='epsilon for ratio clip' )
parser.add_argument('-l2', '--lambd', type=float,help='lambda parameter for GAE' )
args = parser.parse_args()

n_episode = args.n_episode
path = args.path
lr1 = args.learning_rate1
lr2 = args.learning_rate2
lambd = args.lambd
learning_time = args.learning_time
beta = args.beta
eps = args.eps
std = args.std

env = UnityEnvironment(file_name=path)

agent = PPO_Agent(env=env, lr1=lr1, lr2=lr2, lambd=lambd, beta=beta, learning_time=learning_time, std=std, eps=eps)

agent.train(n_episode)