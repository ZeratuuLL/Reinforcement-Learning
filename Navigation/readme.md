This is a small project which uses an simplified version of the Banana Collector on [Unity ML-Agents Giuhub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). They are **NOT** identical.

Here there is only one agent navigating itself towards yellow bananas, which give +1 reward, an also avoiding blue ones, which give -1 reward. The agent is allowed for in total 300 actions. Including moving forward, backward and turning left and right.

In this implement I did not use the raw pixel inputs. Instead I used a 37-dimension state information provided by the environment, containing the agent's velocity, along with ray-based perception of objects around the agent's forward direction. The agent learns to get higher reward with this information. This problem is considered solved if the agent could get an average reward larger than 13 over a consecutive 100 episodes.

To compare the effectiveness and robustness for each algorithm, I tuned the hyperparameters in a small range (should not be bad) and let each run 15 times, with each time lasting 1000 episodes and record their means of the past 100 episodes (called moving average later) as well as their episodic rewards. I visualize the trajectory of means, confidence interval as well as upper and lower bound for both moving average and episodic rewards. Hopefully this can provide a better understanding of the whole picture.

In this experiment we can see (some results to be filled). I also explored these algorithms on LunarLander-v2 environment in gym, which is implemented with tensorflow instead of pytorch. The results seems to be somehow different. You can check it in the other directory.

For downloading the agents, here are the options:
  * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  * [MaxOSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  * [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  * [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
  * [Headless mode](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip), which you will need if you want to train on a remote server

Here is a list for all files in this project:
  * Preparation.md : 
  
      all necessary preparations for this project.
  * Navigation.ipynb :
  
      The jupyter notebook file which provides an example of using this environment. You can also watch the performance of a trained agent with it.
 
  * Navigation_train_DQN.ipynb :
  
      The jupyter notebook file I actually used to train the DQN agent. It also contains the code for training with each algorithm
      
  * Navigation_train_PPO.ipynb :
  
      The jupyter notebook file I actually used to train the PPO agent. It used two methods to calculate the expected reward. It seems that TD is more stable than MC.
      
  * Navigation_train_SQN.ipynb:
  
      The jupyter notebook file I actually used to train the SQN agent. You can check the reference [here](https://arxiv.org/pdf/1702.08165.pdf)
      
  * Report.pdf :
  
      A report of my experiments. Mostly about implement of different DQNs.
      
  * Checkpoints:
  
      A folder saving different trained weights for multiple agents. Now I have commented the line which saves the agents' weights. More details inside.
