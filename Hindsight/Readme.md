This repository contains some experiments with [Hindsight Experience Replay(HER)](https://arxiv.org/abs/1707.01495). You can check more details in the jupyter notebook

## Motivation

Before introducing the idea of HER, we think about such a case: an agent moves in a grid world where states $s\in S=\{0, 1\}^n$. Here $n$ is some positive integer. At every moment the agent knows where it is. It also has a goal to reach. The reward is $1$ if the agent reaches the goal and $-0.1$ for all other steps. Every step the agent can only move to an adjacent grid: which means choosing a coordinate and change it's number. 

When $n$ is small this is quite easy. But when $n$ grows large, random exploration will not help solving this problem. The reason is that the state space is too large to explore and with probability almost equal to 1, all rewards received should be -0.1 and the agent cannot learn any meaningful actions. What should we do to deal with this? Here are some potential answers:

  * Enhance exploration by some methods such as bringing in Curiosity. For example [this paper](https://pathak22.github.io/noreward-rl/resources/icml17.pdf). In the future I will check the efficiency of this
  * Use model based algorithms. This makes the problem a search problem, given the assumption that the model generalizes well to unseen states.
  * Add special 'fake' but 'valid' experiences and learn from them with offline learning methods. This is the idea of **hindsight experience replay**. For more details you can check in the original paper by OpenAI.
  
## Algorithm
