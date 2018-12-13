Files in this repository are related to two Unity environments. [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) and [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler)

## Reacher

For **Reacher** we have two versions. Version 1 has only 1 agent and version 2 has 20. All the agents are exactly the same. They have a double-jointed which can move to target locations. And the goal is to move it's 'hand' to the goal location, and keep it there. The goal location moves continuously around the agent with some constant speed randomly set at the beginning of each episode. The reward is strange, but most time when the 'hand' is in the target area, it wil receive a reward +0.04, also with a small amount of +0.01, +0.02 and +0.03. Each episode there are 1000 steps. The benchmark for this environment is 30.

### Download Environment

Version 1:
  * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
  * [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
  * [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
  * [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
  * [Headless mode](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip), which is necessary if the training is on a remote server

Version 2:
  * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  * [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  * [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  * [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
  * [Headless mode](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip), which is necessary if the training is on a remote server
  
### Preparations

To watch the performance of this agent, please go into **DDPG** folder and read the detailed instructions in **show.ipynb**. That notebook should help you see the skills of a trained agent.

To train the agent yourself, please first make sure you have the following files:
  * the environment file (see links above)
  * all .py files in this page
Then read the explanation of parameters in **DDPG.py** to understand what command you should use to train the agent. For me I used
```python
python './Reinforcement_Learning/new_Continuous_Control/DDPG.py' -p './envs/Reacher-20/Reacher' -c 2 -s1 1 -s2 1 -s3 20 -l 10 -lr1 0.0001 -lr2 0.001 -b 128 -t 0.001 -n 1500
```
to train on my remote server.

## Crawler

Later I will explain this when I get to work on this.

## Folders

In the **DDPG** folder you can find the some plots and a report which should help you understand the result and my findings in implementing DDPG algorithm.

## .py files
  * **infrasturctures.py** contains some basic ingredients for the experiment, including networks, noise generator, replaybuffer and so on
  * **DDPG.py** is the file you can use to train the agent. Please pass in parameters follow the descriptions in the file
  * **agents.py** contains codes for different kind of agents. The agents are defined as class. They can initiailze themselves, take actions, learn from past experiments and train themselves. For now there is only a DDPG agent. 
