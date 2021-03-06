This folder saves the files related to training Crawler with a **fixed** goal. The benchmark given by Unity is 2000. Currently I am only getting 1700~1800. You can see the rewards for 12 agents in the jupyter notebook. It's very close to the benchmark. It seems that the most influencial hyperparameter is the learning rate. I have noted the learning rate for each checkpoint in their file names. The best result comes from **Checkpoint3**

My agent seems to have learnt the correct way of moving but haven't found a faster way of running. This is quite different from the [illustration provided by Unity](https://www.youtube.com/watch?v=ftLliaeooYI&feature=youtu.be). My guess is that this environment is adjusted somehow by Udacity team (you can see the input dimension is 129, but the [original version](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) is 117).

Besides I want to say thanks to Github user [JacobXPX](https://github.com/JacobXPX/Crawler_using_PPO) for using his network structure.

Here is a quick list of the files:
  * Crawler.ipynb: The first block has everything you need to do training. I have commented the last line, which does the training process. The second block tests it's performance during a 1500-timestep window and the third block tests the performance before the first fall of each agent (12 in total).
  * Crawler_Checkpoint_(number)_(learning rate).pth: Saved weights for the network

Something to remind in the end:
  * For each episode I required 3000 steps to collect data to learn from. If this number is too small the training will be much worse (I tried 1500)
  * During training I test the agent every 50 episodes and roll back to the saved one if the current one cannot out-perform it. I test the agent for 1500 timesteps and repeated 20 times for an average.
  * Somehow the whole environment crushed some time during training so I don't have the saved rewards history nor know how many episodes it takes to arrive this. But from the recorded time it should be around 700-800 episodes.
  * During training sometimes you will receive ```nan``` rewards. I do not know the reason behind this. But I am sure it's not due to ```nan``` in actions since there was none. To deal with that everystep I checked the reward and replace all ```nan``` by -5 so that the agent will learn to avoid that.
