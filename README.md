# Reinforcement-Learning
This is a repository where I save my own explorations of reinforcement learning. Here are the brief introductions for the projects.
  * Navigation: A robot walks in a square world collecting yellow banana whose reward is +1 and avoiding blue banana whose reward is -1. Implemented algorithms are original DQN (Deep Q-Learning), Dual-DQN and Prioritized DQN. This is also a part of the [Deep Reinforcement Learning NanoDegree of Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
  * Continuous Control: There are three different environments. One thing common among them is that their action spaces are continuous so DQN cannot be applied directly. Please find more detailed information inside this folder. This is also a part of the [Deep Reinforcement Learning NanoDegree of Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
  * Collaboration and Competition: Two agents playing table tennis in a simplified version where hitting the ball over the net gives reward +0.5 and failing to keep the ball alive will get -1. This is a situation where multiple agents learns to collaborate/compete with each other. I just solved this with DDPG instead of some algorithms for multiple agents. This is also a part of the [Deep Reinforcement Learning NanoDegree of Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
  * Hindsight Experience Replay: A simple illustration of the idea of hindsight experience replay proposed by OpenAI(ClosedAI?) This helps to deal with the situation where agent tries to reach a given goal but the space is too large to explore. Please check other details inside.
  * LunarLander-v2: A repository where I tried to implement some RL Agents based on tensorflow. I tested DQN, SQN, PPO and TD3 on the gym LunarLander-v2 environment. Compare the results with this experiment with the results from navigation project, we can see different algorithms (agents) take advantage of different tasks.
