This repository stores my tests of three different algorithms: DQN (including Double-DQN and Dual-DQN), PPO and SQN on a simple environment: LunarLander-v2 from gym. You can read the official introduction of it [here](https://gym.openai.com/envs/LunarLander-v2/)

The agents are implemented by both tensorflow and keras, but in the tensorflow version I still used keras to create a simple feed forward network. This is the first time I try tensorflow/keras so the syntax might not be clean. Please advise if you can.

For each algorithm I tuned the hyperparameters within a relative small range (which should not be bad) and for each setting I tested 15 runs, with each run lasting 2000 episodes. I recorded the means of past 100 episodes (called moving average later), episodic rewards of 2000 episodes as well the loss encountered during training. I visualized the trajectory of means, confidence interval and lower, upper bound for both moving average and episodic reward. 

From the results we can see that in this task SQN is not really better than DQN given this length of training. This is different from the Navigation experiments. Some discussion about SQN and DQN are also written at the end of the notebook.
