import torch
import torch.nn
import torch.nn.functional as F
from infrastructures import Actor, Critic1, Critic2, ReplayBuffer, OUNoise, device
import torch.optim as optim
import numpy as np
import random
from collections import deque

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
        if critic==1:
            self.critic_local = Critic1(self.state_size, self.action_size).to(device)
            self.critic_target = Critic1(self.state_size, self.action_size).to(device)
        else:
            self.critic_local = Critic2(self.state_size, self.action_size).to(device)
            self.critic_target = Critic2(self.state_size, self.action_size).to(device)
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr1)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr2)
        self.memory = ReplayBuffer(self.action_size, buffer_size=int(1e6), batch_size=self.batch_size,\
                                   seed=random.randint(1, self.batch_size))
        self.noise = OUNoise(size=self.action_size, seed=random.randint(1, self.batch_size))
        
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).detach().cpu().numpy()
        self.actor_local.train()
        noise = self.noise.sample()
        action += noise
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
        
        expected_rewards = scores + (1-dones)*self.gamma*self.critic_target(next_states, self.actor_target(states))
        
        for _ in range(self.speed1):
            observed_rewards = self.critic_local(states, actions)
            L = F.mse_loss(expected_rewards, observed_rewards)
            self.critic_optimizer.zero_grad()
            L.backward()
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
            actions = self.act(states)
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
                actions = self.act(states)
            score_window.append(np.mean(episodic_reward))
            rewards.append(episodic_reward)
            
            print('\rEpisode {}. Total score for this episode: {:.4f}, average score {:.4f}'.format(i, np.mean(episodic_reward),np.mean(score_window)),end='')
            if i % 100 == 0:
                print('\n')
                self.actor_local.cpu()
                self.critic_local.cpu()
                self.actor_target.cpu()
                self.critic_target.cpu()
                torch.save(self.actor_local.state_dict(),'./offline/offline_actor_checkpoint_{}.pth'.format(i))
                torch.save(self.critic_local.state_dict(),'./offline/offline_critic_checkpoint_{}.pth'.format(i))
                torch.save(self.actor_target.state_dict(),'./offline/offline_actor_target_checkpoint_{}.pth'.format(i))
                torch.save(self.critic_target.state_dict(),'./offline/offline_critic_target_checkpoint_{}.pth'.format(i))
                self.actor_local.to(device)
                self.critic_local.to(device)
                self.actor_target.to(device)
                self.critic_target.to(device)
        
        np.save('./offline/offline_rewards.npy',np.array(rewards))
            