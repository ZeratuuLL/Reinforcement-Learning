import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import numpy as np
import random
import copy

gpu=input('Please assing gpu number: ')
device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
print('Using device {}'.format(device))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        #Extract information from memory unit and return
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)    
    
class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, hidden=[400, 300]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        #self.initialize()
        
    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.tanh(self.fc3(x))
        return x
    
    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
        
class Critic(nn.Module):
    
    def __init__(self, state_size, action_size, hidden=[400, 300]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0]+action_size, hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        #self.initialize()
    
    def forward(self, state, action):
        x = self.bn1(F.relu(self.fc1(state)))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.uniform_(-3e-4, 3e-4)
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, agent_num, action_size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones((agent_num, action_size))
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.shape = (agent_num, action_size)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(self.shape[0]*self.shape[1])]).reshape(self.shape)
        self.state = x + dx
        return self.state
    
class FullyConnectedNetwork(nn.Module):
    
    def __init__(self, state_size, output_size, hidden_size, output_gate=None):
        super(FullyConnectedNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.output_gate = output_gate

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x

#######################################################################
# For the following two classes                                       #
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
class PPOPolicyNetwork(nn.Module):
    
    def __init__(self, config):
        super(PPOPolicyNetwork, self).__init__()
        state_size = config['environment']['state_size']
        action_size = config['environment']['action_size']
        hidden_size = config['hyperparameters']['hidden_size']
        self.device = config['pytorch']['device']

        self.actor_body = FullyConnectedNetwork(state_size, action_size, hidden_size, F.tanh)
        self.critic_body = FullyConnectedNetwork(state_size, 1, hidden_size)  
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(self.device)

    def forward(self, obs, action=None):
        obs = torch.Tensor(obs).to(self.device)
        a = self.actor_body(obs)
        v = self.critic_body(obs)
        
        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v