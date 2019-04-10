This folder contains the result for implementation of [PPO](https://arxiv.org/abs/1707.06347) algorothm.

You can have a quick look of the trajectory of episodic reward (the average reward of 20 agents in **rewards.png**. In the jupyter notebook you can see the different settings for updates. The main difference is batchsize and methods to standardize rewards.  There is a clear difference.

If you want a detailed report, please read Reacher_Report.pdf in the outer folder

And the ppo_checkpoint.pth saves the weights for a trained agent. You can follow the instruction in show.ipynb to load it and see the performance of the agents.
