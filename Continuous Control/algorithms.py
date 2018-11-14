import numpy as np
from collections import deque
from Policy_Agents import DDPG_Agent
import torch
from infrastructure import device

def ddpg(N, env, n_episodes, speed1, speed2, steps, learning_time, batch_size):
    '''Use DDPG algorithm to solve the environment
    This function will not learn from the last N timestamps of each episode
    Here we assume num_agents=1
    
    Params
    ======
    N (int): N-step Sarsa
    env: the unity environment
    n_episodes (int): max number of episodes for training
    '''
    
    #Some hyper parameters
    LR = 0.0001
    Gamma = 0.99
    Tau = 0.001
    Batch_size = batch_size
    
    #initialize Brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    
    #Initizalize Agent
    agent = DDPG_Agent(state_size, action_size, learning_rate=LR, gamma=Gamma, tau=Tau, \
                       batch_size=Batch_size, speed1=speed1, speed2=speed2, num_agents=num_agents)      # Create agent 
    rewards = []                                                   # Record rewards from all episodes
    window = deque(maxlen=100)
    discount = Gamma**np.arange(N+1)
        
    for i in range(1, n_episodes+1):
        
        t = 0                                                      # Record how many steps has gone in this episode
        reward_history = np.zeros((N+1,num_agents))                # Record rewards history within this episode
        states_history = np.zeros((N+1, num_agents, state_size))   # Record states history within this episode
        actions_history = np.zeros((N+1, num_agents, action_size)) # Record rewards actions within this episode
            
        # reset The environment & save start information
        env_info = env.reset(train_mode=True)[brain_name]        
        state = env_info.vector_observations
        action = agent.act(state)
        scores = np.zeros(num_agents)
        states_history[-1,:,:] = state.copy()
        actions_history[-1,:,:] = action.detach().numpy().copy()
        
        while True:
            #Get new information about the environment
            env_info = env.step(action.detach().numpy())[brain_name]
            next_state = env_info.vector_observations
            reward = np.array(env_info.rewards)
            done = np.array(env_info.local_done).reshape(num_agents,-1)
            #Update reward history
            reward_history[:-1,:] = reward_history[1:,:]
            reward_history[-1,:] = reward
            scores += reward
            
            #Check whether to save memory
            t += 1
            if t>=N+1: #Have at least N+1 steps now
                # Save experience into agent's memory
                temp_rewards = reward_history*discount[:,np.newaxis]
                temp_rewards = np.sum(temp_rewards, axis=0).reshape(num_agents,-1)
                for exp in zip(states_history[0,:].copy(), actions_history[0,:].copy(), temp_rewards, next_state, done):
                        agent.memory.add(exp[0], exp[1], exp[2], exp[3], exp[4])
            
            if len(agent.memory.memory)>=Batch_size:
                if t % steps == 0:
                    for j in range(learning_time):
                        agent.learn(N)
                
            #Save new record
            next_action = agent.act(next_state)
            states_history[:-1,:,:] = states_history[1:,:,:]
            actions_history[:-1,:,:] = actions_history[1:,:,:]
            states_history[-1,:,:] = next_state.copy()
            actions_history[-1,:,:] = next_action.detach().numpy().copy()
            action = next_action
            if any(done):
                break
        
        rewards.append(scores)
        window.append(scores)
        print('\rEpisode {}\t Score of this episode: {:.4f}\tAverage Score: {:.4f}'.format(i, np.mean(scores), np.mean(window)), end='')
        if i % 50 ==0:
            print('\nEpisode {}\tAverage Score: {:.4f}'.format(i, np.mean(window)))
        if (np.mean(window)>=30) & (i>=100):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(i-100, np.mean(window)))
            torch.save(agent.actor_local.state_dict(),'actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(),'critic_checkpoint.pth')
            break
        if i % 10 ==0:
            agent.actor_local.cpu()
            agent.critic_local.cpu()
            torch.save(agent.actor_local.state_dict(),'actor_checkpoint_{}.pth'.format(i))
            torch.save(agent.critic_local.state_dict(),'critic_checkpoint_{}.pth'.format(i))
            agent.actor_local.to(device)
            agent.critic_local.to(device)
            
    print('\nEnvironment not solved!\tAverage Score: {:.4f}'.format(np.mean(np.array(window))))
    agent.actor_local.cpu()
    agent.critic_local.cpu()
    agent.actor_target.cpu()
    agent.critic_target.cpu()
    torch.save(agent.actor_local.state_dict(),'actor_checkpoint.pth')
    torch.save(agent.critic_local.state_dict(),'critic_checkpoint.pth')
    torch.save(agent.actor_local.state_dict(),'actor_target_checkpoint.pth')
    torch.save(agent.critic_local.state_dict(),'critic_target_checkpoint.pth')
    return rewards