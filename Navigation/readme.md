This is a small project which uses an simplified version of the Banana Collector on [Unity ML-Agents Giuhub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). They are NOT identical.

Here there is only one agent navigating itself towards yellow bananas, which give +1 reward, an also avoiding blue ones, which give -1 reward. The agent is allowed for in total 300 actions. Including moving forward, backward and turning left and right.

In this implement I did not use the raw pixel inputs. Instead I used a 37-dimension state information provided by the environment, containing the agent's velocity, along with ray-based perception of objects around the agent's forward direction. The agent learns to get higher reward with this information. This problem is considered solved if the agent could get an average reward larger than 13 over a consecutive 100 episodes.

Here is a list for all files in this project:
  * Preparation.md : 
  
      all necessary preparations for this project.
  * Navigation.ipynb :
  
      The jupyter notebook file which provides an example of using this environment. You can also watch the performance of a trained agent with it.
  * RL_Agent.py : 
  
      The .py file which contains the code for the agent.
  * Net.py : 
  
      The .py file which contains the neural network structure.
  * : chechpoint.pth
  
      The .pth file (created by ```torch.save()```) which saves the weights of a trained agent. Not using the Dual structure
  * train.py : 
  
      The .py file which contains the function that can be used to train the agent.
  * Navigation_train.ipynb :
  
      The jupyter notebook file I actually used to train the agent. It also contains the code for training with each algorithm
  * Report.pdf :
  
      A report of my experiments. Still updating.
