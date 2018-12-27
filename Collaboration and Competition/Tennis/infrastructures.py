import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import numpy as np
import random
import copy

gpu = input('Please assign gpu: ')
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
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
        self.initialize()
        
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
        
class Critic1(nn.Module):
    
    def __init__(self, state_size, action_size, hidden=[400, 300]):
        super(Critic1, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0]+action_size, hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.initialize()
    
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
        
class Critic2(nn.Module):
    
    def __init__(self, state_size, action_size, hidden=[400, 300]):
        super(Critic2, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1]+action_size, 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.initialize()
    
    def forward(self, state, action):
        x = self.bn1(F.relu(self.fc1(state)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = torch.cat([x, action], dim=1)
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