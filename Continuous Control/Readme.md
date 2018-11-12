Files in this repository are related to two Unity environments. [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) and [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler)

For **Reacher** we have two versions. Version 1 has only 1 agent and version 2 has 20. All the agents are exactly the same. They have a double-jointed which can move to target locations. And the goal is to move it's hand to the goal location, and keep it there. The goal location moves continuously around the agent. Every agent receives +0.1 reward each step agent's hand is in goal location. The benchmark is 30

For **Crawler** 
Later I will explain this when I get to work on this.

Now I have uploaded the files for [DDPG](https://arxiv.org/abs/1509.02971). For now it does not work. I am not sure where I made a mistake. The learning part and the saving experience part should be correct. So is that my network design? Or the hypermaters? Anyway it's not learning the correct thing. If anyone happens to see this, it would be very nice if you could help me find the bugs.

Files:
  * **infrastructure.py**: where I put the codes for networks and replaybuffer
  * **Policy_Agents**: where I put the codes for agents
  * **algorithms.py**: where I save different algorithms
  * **Reacher-1.py**: Initiate the environment, create agent and run the algorithm
