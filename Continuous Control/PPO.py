import torch
import torch.optim as optim
from unityagents import UnityEnvironment

from infrastructures import PPOPolicyNetwork
from agents import PPOAgent

env = UnityEnvironment(file_name="./envs/Reacher-20/Reacher")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

config = {
    'environment': {
        'state_size':  env_info.vector_observations.shape[1],
        'action_size': brain.vector_action_space_size,
        'number_of_agents': len(env_info.agents)
    },
    'pytorch': {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    },
    'hyperparameters': {
        'discount_rate': 0.99,
        'tau': 0.95,
        'gradient_clip': 5,
        'rollout_length': 2048,
        'optimization_epochs': 10,
        'ppo_clip': 0.2,
        'log_interval': 2048,
        'max_steps': 1e5,
        'mini_batch_number': 32,
        'entropy_coefficent': 0.01,
        'episode_count': 250,
        'hidden_size': 512,
        'adam_learning_rate': 3e-4,
        'adam_epsilon': 1e-5
    }
}
    
policy = PPOPolicyNetwork(config)
optimizier = optim.Adam(policy.parameters(), config['hyperparameters']['adam_learning_rate'], eps=config['hyperparameters']['adam_epsilon'])
agent = PPOAgent(env, brain_name, policy, optimizier, config)
agent.train(1500)