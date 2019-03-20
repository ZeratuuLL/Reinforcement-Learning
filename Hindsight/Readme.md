This repository contains some experiments with [Hindsight Experience Replay(HER)](https://arxiv.org/abs/1707.01495). You can check more details in the jupyter notebook

## Motivation

Before introducing the idea of HER, we think about such a case: an agent moves in a grid world where states $s\in S=\{0, 1\}^n$. Here $n$ is some positive integer. At every moment the agent knows where it is. It also has a goal to reach. The reward is $1$ if the agent reaches the goal and $-0.1$ for all other steps. Every step the agent can only move to an adjacent grid: which means choosing a coordinate and change it's number. 

When $n$ is small this is quite easy. But when $n$ grows large, random exploration will not help solving this problem. The reason is that the state space is too large to explore and with probability almost equal to 1, all rewards received should be -0.1 and the agent cannot learn any meaningful actions. What should we do to deal with this? Here are some potential answers:

  * Enhance exploration by some methods such as bringing in Curiosity. For example [this paper](https://pathak22.github.io/noreward-rl/resources/icml17.pdf). In the future I will check the efficiency of this
  * Use model based algorithms. This makes the problem a search problem, given the assumption that the model generalizes well to unseen states.
  * Add special 'fake' but 'valid' experiences and learn from them with offline learning methods. This is the idea of **hindsight experience replay**. For more details you can check in the [original paper by OpenAI](https://arxiv.org/abs/1707.01495).
  
## Algorithm

So how should we add fake but valid experiences? We should create transitions will various reward signals and guarantee that they still follow the dynamic of the environment. To achieve this we must know the true reward function. Luckily this is usually true for lower level worker in hierarchical reinforcement learning. But we don't need the transition dynamic because we can make use of the real experiences. The only thing we modify is the goal in experiences and the reward accordingly. 

In the paper, it's mentioned that we can have a function to generate a set of 'additional goals' based on any experiences from an episode. Then we will create a new but illusional episode by replacing the goal with one goal in 'additional goals' we generated. These new experiences will be added into replay buffer and helps training. If you wonder how to generate 'additional' goals, the simplest method is to return the final state as an additional goal

In the jupyter notebook, I implemented this in a slightly different way. Say in an episode we have state trajectory S(1), S(2),... S(n). My implement chooses some time between 1 to n, which we can call t(1)=1, t(2), ... t(k)=n. For each i from {2, 3, ..., n}, I set the goals from time t(i-1) to t(i) as s(i) and change the rewards accordingly. This will guarantee the new experience has k successful trajectories. The proportion k/n is a parameter that we can choose.

For more details about the environment and experiment results please check the jupyter notebook.

## Future

Now I am working on extending this method to PPO. The authors HER did not mention this in their paper since PPO does not make use of a replay buffer explicitly. But actually for each episode PPO have a temporary replay buffer. I will update this part when time comes
