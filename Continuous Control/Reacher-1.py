from unityagents import UnityEnvironment
from algorithms import ddpg
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parameters for Continuous Control')
parser.add_argument('-e', '--episode', type=int,help='Input the number of episodes you want')
parser.add_argument('-n', '--N', type=int,help='Decide the steps for N-step MC estimate' )
parser.add_argument('-p', '--path', type=str,help='the path of environment file' )
args = parser.parse_args()

N = args.N
n_episodes = args.episode
path = args.path
print('the path is: \''+path+'\'')

env = UnityEnvironment(file_name=path)

rewards = ddpg(N, env, n_episodes)

np.save('Reacher-1.npy',rewards)