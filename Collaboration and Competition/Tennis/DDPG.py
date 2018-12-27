from unityagents import UnityEnvironment
from agents import DDPG_Agent
import argparse

parser = argparse.ArgumentParser(description='Parameters for Continuous Control (DDPG Agent)')
parser.add_argument('-n', '--n_episode', type=int,help='Input the number of episodes you want')
parser.add_argument('-p', '--path', type=str,help='the path of environment file' )
parser.add_argument('-s', '--steps', type=int,help='Gap between two times of learning behaviors' )
parser.add_argument('-l', '--learning_time', type=int,help='How many times to learn in each learning behavior' )
parser.add_argument('-lr1', '--learning_rate1', type=float,help='learning rate for actor' )
parser.add_argument('-lr2', '--learning_rate2', type=float,help='learning rate for critic' )
parser.add_argument('-b', '--batch_size', type=int,help='Decide the batch size' )
parser.add_argument('-t', '--tau', type=float,help='soft update parameter' )
args = parser.parse_args()

n_episode = args.n_episode
path = args.path
lr1 = args.learning_rate1
lr2 = args.learning_rate2
tau = args.tau
step = args.steps
learning_time = args.learning_time
batch_size = args.batch_size

env = UnityEnvironment(file_name=path)

for i in range(1, 11):
    agent = DDPG_Agent(env, lr1, lr2, tau, step, learning_time, batch_size)
    agent.train(n_episode, i)