import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from infrastructure import Actor, Critic, ReplayBuffer, device

print('device in Policy_Agents.py is {}'.format(device))

class DDPG_Agent():
    '''Creates a DDPG agent that interacts with and learn from the environment'''
    
    def __init__(self, state_size, action_size, learning_rate, gamma, tau, batch_size, epsilon):
        '''Initialize the agent.
        
        Params
        ======
        state_size (int): the length of state size
        action_size (int): the length of action_size
        learning_rate (float): learning rate for the networks
        gamma (float): reward discount rate
        tau (float): parameter for soft update
        batch_size (int): batch size for learning
        epsilon (float): standard deviation for the noise when choosing next action 
        '''
        #basic parameters
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.step = 0
        
        #networks & optimizers
        self.actor_local = Actor(self.state_size, self.action_size, policy=False).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, policy=False).to(device)
        self.critic_local = Critic(self.state_size, self.action_size, action=True).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, action=True).to(device)
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic_local.parameters(), lr=self.lr)
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)
        
        #replay buffer
        self.memory = ReplayBuffer(batch_size=self.batch_size, seed=1)
    
    def act(self, state):
        '''Use current actor network to take action. with a random noise'''
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(torch.tensor(state, dtype=torch.float).to(device)).cpu()
        self.actor_local.train()
        noise = np.random.standard_normal(action.detach().numpy().shape)*self.epsilon
        action = action + torch.tensor(noise, dtype=torch.float)
        action = torch.clamp(action, min=-1, max=1)
        return action
    
    def learn(self, n):
        '''Train local networks use n-step MC, The n-step reward is already calculated and saved in memory
        
        Params
        ======
        n (int): The number of steps of MC
        '''
        self.step += 1
        
        #Some calculation
        experiences = self.memory.sample()
        states, actions, rewards, target_states, dones = experiences #rewards here are n-step rewards
        pred_actions = self.actor_target(target_states)
        pred_actions = torch.clamp(pred_actions, min=-1, max=1)
        observed_rewards = self.critic_target(torch.cat([target_states, pred_actions], dim=1))*self.gamma**(n+1)
        observed_rewards = (1-dones)*observed_rewards + rewards
        expected_rewards = self.critic_local(torch.cat([states, actions], dim=1))
        
        #Update critic
        cirtic_loss = F.mse_loss(observed_rewards, expected_rewards)
        self.optimizer_critic.zero_grad()
        cirtic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.optimizer_critic.step()
        
        #Update actor
        pred_actions = self.actor_local(states)
        pred_actions = torch.clamp(pred_actions, min=-1, max=1)
        actor_loss = -self.critic_local(torch.cat([states, pred_actions], dim=1)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actor_local.parameters(), 1)
        self.optimizer_actor.step()
        
        #Update target networks
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        
        #Decrease exploration
        self.epsilon *= 0.9995
        
    def soft_update(self, local_net, target_net, tau):
        '''Do soft update to target_net'''
        
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)