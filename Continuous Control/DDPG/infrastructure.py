import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

BATCH_SIZE=64
gpu = input('Please assign the gpu number: ')
device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))

def hidden_init(layer):
    fan_in = layer.weight.data.size()[-1]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, batch_size, seed, buffer_size=int(1e5)):
        """Initialize a ReplayBuffer object.

        Params
        ======
            #action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        #self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "target_state",\
                                                                "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, target_state, done):
        """Add a new experience to memory.
        
        Params
        ======
        state: the 'current' state
        action: the 'current' action
        reward: the processed n_step MC reward
        target_state: the 'target' state which we would apply bootstrap to estimate value
        target_action: the action taken at target_state, will only be used when use Sarsa to calculate 
                       advantage function. Necessary for some continuous cases like DDPG
        done: If the current state and action ends the episode
        """
        
        e = self.experience(state, action, reward, target_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory
        
        """
        experiences = random.sample(self.memory, self.batch_size)
        
        #Extract information from memory unit and return
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        target_states = torch.from_numpy(np.vstack([e.target_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, target_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Actor(nn.Module):
    '''Network structure for agent's actor'''
    
    def __init__(self, state_size, action_size, hidden=[400, 300], policy=True):
        '''Build Structure
        
        Params
        ======
        state_size (int): state size of the environment
        action_size (int): action size of the environment
        policy (bool): if True, the output will be a softmax layer, otherwise it would be a fully connected layer
        '''
        super(Actor, self).__init__()
        dims = [state_size] + hidden + [action_size]
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.out = nn.Softmax(dim=1)
        self.policy = policy
        self.reset_parameters()
        
    def forward(self, x):
        '''Feed forward'''
        
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        if self.policy:
            return self.out(x)
        else:
            return F.tanh(x)
        
    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
            layer.bias.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.layers[-1].bias.data.uniform_(-3e-3, 3e-3)
    
class Critic(nn.Module):
    '''Network structure for agent's critic'''
    def __init__(self, state_size, action_size, hidden=[400, 300], action=False):
        '''Build Structure
        
        Params
        ======
        state_size (int): state size of the environment
        action_size (int): action size of the environment
        action (bool): if True, the critic takes action as a part of input and return Q(s,a), otherwise returns V(s)
        '''
        super(Critic, self).__init__()
        dim = state_size
        dims = [dim] + hidden + [1]
        Ins = dims[:-1]
        Outs = dims[1:]
        Ins[1] += action_size # Insert actions here, only affect input dimension
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(Ins, Outs)])
        self.reset_parameters()
    
    def forward(self, states, actions):
        '''Feed forward'''
        x = states
        for layer in self.layers[:1]:
            x = F.relu(layer(x))
        x = torch.cat([x, actions],dim=1)
        for layer in self.layers[1:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    
    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
            layer.bias.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-4, 3e-4)
        self.layers[-1].bias.data.uniform_(-3e-4, 3e-4)
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state