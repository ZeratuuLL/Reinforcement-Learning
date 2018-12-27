This repository contains file realted two environments. [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) and Soccer in [this page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). Notice that they are not identical but mostly the same.

To run the python source codes or jupyter notebook, you need to first set up an environment because it uses an older edition Unity API. You can follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).

## Tennis
### Environment Introduction
In this environment there are two agents. Each of them controls a racket to hit the ball over a net. If the ball passes the net, the agent which hits it will get reward +0.1. But if the ball falls on the ground, the agent which fails to hit it over the net will receive loss -0.01. Once the ball hits the ground, or a certain length of time has passed, an episode ends and the final score is the average score of two agents. The benchmark here is 0.5 but the perfect solution should be 2.6. If you have an average score over 100 consecutive episodes beyond the benchmark, this environment is considered solved.

### Environment Files
In the links below you can find the environment for different systems.
  * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  * [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  * [Windows (32 bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  * [Windows (64 bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
  * [Headless Mode](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip), which you will need to train this on a remote server.
  
### Algorithms
For now I have an implementation of [DDPG](https://arxiv.org/abs/1509.02971) algorithm and working on [MADDPG](https://arxiv.org/abs/1706.02275). For more details please go into the foler **Tennis**

## Soccer
I will complete this part after solving this environment.
