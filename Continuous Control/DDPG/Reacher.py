from unityagents import UnityEnvironment
from algorithms import ddpg
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parameters for Continuous Control')
parser.add_argument('-e', '--episode', type=int,help='Input the number of episodes you want')
parser.add_argument('-n', '--N', type=int,help='Decide the steps for N-step MC estimate' )
parser.add_argument('-p', '--path', type=str,help='the path of environment file' )
parser.add_argument('-s1', '--speed1', type=int,help='How many times slower to do soft update' )
parser.add_argument('-s2', '--speed2', type=int,help='How many times slower to do soft update' )
parser.add_argument('-s3', '--steps', type=int,help='Gap between two times of learning behaviors' )
parser.add_argument('-l', '--learning_time', type=int,help='How many times to learn in each learning behavior' )
parser.add_argument('-lr', '--learning_rate', type=float,help='learning rate' )
parser.add_argument('-b', '--batch_size', type=int,help='Decide the batch size' )
parser.add_argument('-t', '--tau', type=float,help='soft update parameter' )
args = parser.parse_args()

N = args.N
n_episodes = args.episode
path = args.path
speed1 = args.speed1
speed2 = args.speed2
steps = args.steps
learning_time = args.learning_time
batch_size = args.batch_size
lr = args.learning_rate
tau = args.tau
print('the path is: \''+path+'\'')

env = UnityEnvironment(file_name=path)

rewards = ddpg(N, env, n_episodes, speed1, speed2, steps, learning_time, batch_size, lr, tau)

np.save('Reacher-1.npy',rewards)