import torch
import torch.nn
import torch.nn.functional as F
from infrastructures import Actor, Critic, ReplayBuffer, OUNoise, device
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from torch.utils.data import TensorDataset, DataLoader


###############################################
#                                             #
#                                             #
#                 DDPG Agent                  #
#                                             #
#                                             #
###############################################   

class DDPG_Agent:
    
    def __init__(self, env, critic, lr1=0.0001, lr2=0.001, tau=0.001, speed1=1, speed2=1,\
                 step=1, learning_time=1, batch_size=64):
        
        #Initialize environment
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        states = env_info.vector_observations
        state_size = states.shape[1]
        self.env = env
        
        #Initialize some hyper parameters of agent
        self.lr1 = lr1
        self.lr2 = lr2
        self.tau = tau
        self.speed1 = speed1
        self.speed2 = speed2
        self.learning_time = learning_time
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.gamma = 0.99
        self.step = step
        
        #Initialize agent (networks, replyabuffer and noise)
        self.actor_local = Actor(self.state_size, self.action_size).to(device)
        self.actor_target = Actor(self.state_size, self.action_size).to(device)
        self.critic_local = Critic(self.state_size, self.action_size).to(device)
        self.critic_target = Critic(self.state_size, self.action_size).to(device)
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr1)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr2)
        self.memory = ReplayBuffer(self.action_size, buffer_size=int(1e6), batch_size=self.batch_size,\
                                   seed=random.randint(1, self.batch_size))
        self.noise = OUNoise(agent_num=self.num_agents, action_size=self.action_size, seed=random.randint(1, self.batch_size))
        
    def act(self, state, i):
        state = torch.tensor(state, dtype=torch.float).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).detach().cpu().numpy()
        self.actor_local.train()
        noise = self.noise.sample()
        action += noise/math.sqrt(i)
        action = np.clip(action, -1, 1)
        return action
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def learn(self):
        
        experiences = self.memory.sample()
        states, actions, scores, next_states, dones = experiences
        
        expected_rewards = scores + (1-dones)*self.gamma*self.critic_target(next_states, self.actor_target(next_states))
        
        for _ in range(self.speed1):
            observed_rewards = self.critic_local(states, actions)
            L = F.mse_loss(expected_rewards, observed_rewards)
            self.critic_optimizer.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()
            del L
        
        for _ in range(self.speed2):
            L = -self.critic_local(states, self.actor_local(states)).mean()
            self.actor_optimizer.zero_grad()
            L.backward()
            self.actor_optimizer.step()
            del L
        
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        
            
    def train(self, n_episodes):
        rewards = []
        brain_name = self.env.brain_names[0]
        score_window = deque(maxlen=100)
        
        for i in range(1, n_episodes+1):
            episodic_reward = np.zeros(self.num_agents)
            env_info = self.env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            actions = self.act(states, i)
            t=0
            
            while True:
                env_info = self.env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                dones = env_info.local_done
                scores = env_info.rewards
                episodic_reward += np.array(scores)
                for state, action, score, next_state, done in zip(states, actions, scores, next_states, dones):
                    self.memory.add(state, action, score, next_state, done)
                t += 1
                
                if len(self.memory.memory)>self.batch_size:
                    if t % self.step == 0:
                        for _ in range(self.learning_time):
                            self.learn()
                
                if any(dones):
                    break
                            
                states = next_states
                actions = self.act(states, i)
            score_window.append(np.mean(episodic_reward))
            rewards.append(episodic_reward)
            
            print('\rEpisode {}. Total score for this episode: {:.4f}, average score {:.4f}'.format(i, np.mean(episodic_reward),np.mean(score_window)),end='')
            if i % 100 == 0:
                print('')
        
        np.save('./offline/offline_rewards.npy',np.array(rewards))
        self.actor_local.cpu()
        self.critic_local.cpu()
        torch.save(self.actor_local.state_dict(),'./offline/actor_checkpoint.pth')
        torch.save(self.critic_local.state_dict(),'./offline/critic_checkpoint.pth')


###############################################
#                                             #
#                                             #
#                  PPO Agent                  #
#                                             #
#                                             #
############################################### 

#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

class PPOAgent(object):
    
    def __init__(self, environment, brain_name, policy_network, optimizier, config):
        self.config = config
        self.hyperparameters = config['hyperparameters']
        self.network = policy_network
        self.optimizier = optimizier
        self.total_steps = 0
        self.all_rewards = np.zeros(config['environment']['number_of_agents'])
        self.episode_rewards = []
        self.environment = environment
        self.brain_name = brain_name
        self.device = config['pytorch']['device']
        self.batch_number = self.hyperparameters['mini_batch_number']
        self.learning_time = self.hyperparameters['optimization_epochs']
        
        self.gamma = self.hyperparameters['discount_rate']
        self.tau = self.hyperparameters['tau']
        self.eps = self.hyperparameters['ppo_clip']

    def learn(self, states, actions, log_probs_old, advantages, returns):
        '''
        Create a dataset with the input data and do several time mini-batch learning
        '''
        mydata = TensorDataset(states, actions, log_probs_old, advantages, returns)
        Loader = DataLoader(mydata, batch_size = states.size(0) // self.batch_number, shuffle = True)
        
        for i in range(self.learning_time):
            for sampled_states, sampled_actions, sampled_log_probs, sampled_advantages, sampled_returns in iter(Loader):
                sampled_advantages = sampled_advantages.to(self.device)
                _, new_log_probs, entropy_loss, values = self.network(sampled_states, sampled_actions.to(self.device))
                ratio = (new_log_probs - sampled_log_probs.to(self.device)).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.eps, 1.0 + self.eps) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0)
                
                value_loss = 0.5 * (sampled_returns.to(self.device) - values).pow(2).mean()
                
                self.optimizier.zero_grad()
                (policy_loss + value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.hyperparameters['gradient_clip'])
                self.optimizier.step()

    def step(self):
        '''
        This function does the following things:
            1. collects the trajectories from one episode with a given length
            2. calculates the advantage functions and the target values for critic
            3. goes into learning step
            4. returns episodic rewards
        '''
        
        #Initialize
        self.episode_rewards = []
        hyperparameters = self.hyperparameters

        env_info = self.environment.reset(train_mode=True)[self.brain_name]    
        states = env_info.vector_observations
        states_history = []
        values_history = []
        actions_history = []
        rewards_history = []
        log_probs_history = []
        dones_history = []
        
        #Collect trajectories
        for _ in range(hyperparameters['rollout_length']):
            actions, log_probs, _, values = self.network(states)
            env_info = self.environment.step(actions.cpu().detach().numpy())[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            terminals = np.array([1 if t else 0 for t in env_info.local_done])
            self.all_rewards += rewards
            
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.all_rewards[i])
                    self.all_rewards[i] = 0
                    
            states_history.append(torch.tensor(states, dtype=torch.float))
            values_history.append(values.detach().cpu())
            actions_history.append(actions.detach().cpu())
            log_probs_history.append(log_probs.detach().cpu())
            rewards_history.append(torch.tensor(rewards, dtype=torch.float))
            dones_history.append(torch.tensor(terminals, dtype=torch.float))                    
            
            states = next_states
        
        #Calculate advantage and returns
        Advantages = []
        advantages = 0
        Returns = []
        returns = 0
        for i in reversed(range(len(states_history) - 1)):
            returns = rewards_history[i] + (1-dones_history[i])*returns*self.gamma
            Returns.append(returns.view(-1))
            states = states_history[i]
            values = values_history[i]
            next_states = states_history[i+1]
            next_values = values_history[i+1]
            delta = rewards_history[i].view(-1,1) + (1-dones_history[i].view(-1,1))*self.gamma*next_values - values
            advantages = advantages*self.gamma*self.tau*(1-dones_history[i].view(-1,1)) + delta
            Advantages.append(advantages.view(-1))
        Advantages.reverse()
        Advantages = torch.stack(Advantages).detach()
        Advantages = Advantages.view(-1,1)
        Returns.reverse()
        Returns = torch.stack(Returns).detach()
        Returns = Returns.view(-1,1)
            
        states = torch.cat(states_history[:-1], 0)
        actions = torch.cat(actions_history[:-1], 0)
        log_probs_old = torch.cat(log_probs_history[:-1], 0)
        advantages = Advantages
        returns = Returns

        advantages = (advantages - advantages.mean()) / advantages.std()
        
        #do the updates
        self.learn(states, actions, log_probs_old, advantages, returns)
        return(np.mean(self.episode_rewards))
        
    def train(self, n_episodes):
        '''
        This function trains the agent with given number of episodes, in each episode:
            calls  agent.step() to conduct learning
            collects episodic rewards and output some information
        Finally returns the rewards data
        '''
        rewards = []
        score_window = deque(maxlen=100)
        for i in range(1, 1+n_episodes):
            episodic_reward = self.step()
            rewards.append(episodic_reward)
            score_window.append(episodic_reward)
            
            print('\rEpisode {}. Total score for this episode: {:.4f}, average score {:.4f}'.format(i, np.mean(episodic_reward),np.mean(score_window)),end='')
            if i % 100 == 0:
                print('')
                
        np.save('offline/rewards_history.npy', np.array(rewards))
        torch.save(self.network.state_dict(),'./offline/PPO_Network.pth')
            
