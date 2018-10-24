import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 5e-4              # learning rate 
UPDATE_EVERY = 4        # how often to update the network
DEFAULT_PRIORITY=1e-2   # initialization value for priority

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, prioritized=False, Dual=False, alpha=0, learning_rate=LR, gamma=GAMMA, tau=TAU):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.lr=learning_rate
        self.tau=tau
        self.gamma=gamma
        self.prioritized=prioritized
        
        # Q-Network
        if Dual:
            self.qnetwork_local = Dual_QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = Dual_QNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.priority=deque(maxlen=BUFFER_SIZE)
        if self.prioritized:
            self.alpha=alpha
        else:
            self.alpha=0# The default value is 0, which is the not prioritized case
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self.priority.append(DEFAULT_PRIORITY)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experience, index = self.memory.sample(self.priority,self.alpha)
                self.learn(experience, index, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, index, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            index : The index of selected experiences. Used to select and update priority
            gamma (float): discount factor
            
        """
        states, actions, rewards, next_states, dones= experiences

        self.optimizer.zero_grad()
        loss=0
        ## TODO: find the correct way to calculate the weights to make the update unbiased.
        ## Should: each experience times 1/(NP_i) to correct the bias introduced by P_i
        ## In the paper: 1/(NP_i)^beta. And normalize the weights so that the largest one is 1 (Not used here)
        
        # ------------------- calculate the weights ------------------- #
        ## If it's prioritized DQN, calculate the weights, otherwise, use 1
        if self.prioritized:
            w=np.array(self.priority)**self.alpha
            w=w/np.sum(w)
            w=w[index]
            w=1/(len(agent.memory)*w)
            w=torch.from_numpy(w)
        else:
            w=torch.ones(BATCH_SIZE)
        
        # ------------------- calculate the update  ------------------- #
        expected_rewards=self.qnetwork_local(states)
        expected_rewards=expected_rewards[range(expected_rewards.shape[0]), actions.cpu().numpy().reshape(-1)]
        expected_next_rewards=self.qnetwork_target(next_states)
        expected_next_rewards,_=torch.max(expected_next_rewards,1)
        real_rewards=rewards.reshape(-1)+self.gamma*(1-dones).reshape(-1)*expected_next_rewards
        w=w.float()
        loss=torch.sum(w*(expected_rewards-real_rewards)**2)/BATCH_SIZE
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
        # -------------------    update priority    ------------------- #
        ## Only update when the method is prioritized DQN
        if self.prioritized:
            priority=np.array(self.priority)
            values=abs((expected_rewards-real_rewards).detach().numpy())
            priority[index]=np.where(values>DEFAULT_PRIORITY,values,DEFAULT_PRIORITY)
            self.priority=deque(priority,maxlen=BUFFER_SIZE)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


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
    
    def sample(self,priority,alpha):
        """Randomly sample a batch of experiences from memory.
        
        Params
        ======
            priority: a vector records the priority for each experience
            alpha: to determine how much the priority should influence the probability. 
                p(sample i) ~ priority(sample i)**alpha
        """
        #Sample experiences with priority. First sample index, then pick corresponding experience
        prob=np.array(priority)**alpha
        prob=prob/np.sum(prob)#This makes it a probability
        index=np.random.choice(a=len(prob),size=self.batch_size,replace=False,p=prob)
        experiences = list(map(self.memory.__getitem__,index))
        
        #Extract information from memory unit and return
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones), index

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)