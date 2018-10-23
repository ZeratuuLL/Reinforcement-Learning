This is a small project which uses an simplified version of the Banana Collector on [Unity ML-Agents Giuhub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). They are NOT identical.

Here there is only one agent navigating itself towards yellow bananas, which give +1 reward, an also avoiding blue ones, which give -1 reward. The agent is allowed for in total 300 actions. Including moving forward, backward and turning left and right.

In this implement I did not use the raw pixel inputs. Instead I used a 37-dimension state information provided by the environment, containing the agent's velocity, along with ray-based perception of objects around the agent's forward direction. The agent learns to get higher reward with this information. This problem is considered solved if the agent could get an average reward larger than 13 over a consecutive 100 episodes.

Here is a list for all files in this project:
  * Preparation.md: all necessary preparations for this project.
  * :The jupyter notebook file which provides an example of using this environment. You can also watch the performance of a trained agent with it.
  * :The .py file which contains the code for the agent.
  * :The .py file which contains the neural network structure.
  * :The .pth file (created by ```torch.save()```) which saves the weights of a trained agent.
  * :The .py file which trains the agent.
  * :The jupyter notebook file I actually used to train the agent.
  * :A summary of my experiments. Still updating.
