This repository stores my tests of three different algorithms: DQN (including Double-DQN and Dual-DQN), PPO and SQN on a simple environment: LunarLander-v2 from gym. You can read the official introduction of it [here](https://gym.openai.com/envs/LunarLander-v2/)

The agents are implemented by tensorflow. This is the first time I try tensorflow so the syntax might be pretty bad. Please advise if you can.

For each algorithm I tuned the hyperparameters within a relative small range (which should not be bad) and for each setting I tested 15 runs, with each run lasting 2000 episodes. I recorded the means of past 100 episodes (called moving average later), episodic rewards of 2000 episodes as well the loss encountered during training. I visualized the trajectory of means, confidence interval and lower, upper bound for both moving average and episodic reward. 

For the results we can see (something to be filled. Tests still running).
