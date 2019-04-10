The jupyter notebook in this folder tries to solve Reacher environment by [Soft Actor Critic(SAC)](https://arxiv.org/abs/1801.01290) method. You can click the link to read the original paper. The general idea is below:

## Short Introduction
Unlike most model-free methods, which maintains a critic V(s) or Q(s,a) and optimize policy pi so that E[r_t+V(s_{t+1})] or E[r_t+Q(s_{t+1},a_{t+1}=pi(s_{t+1}))] is maximized, SAC takes into entropy into consideration. It also maintains a critic and takes the entropy of next step as an 'extra reward'. To be exact,

Define J(pi) = \sum_t E[r_t+\alpha*H(pi(*|s_{t+1}))]. Here H(P) is the entropy of a distribution P. SAC wants to maximize J(pi) by minimizing D_{KL}(pi(*|s_t)||exp(Q(s_t,*)/alpha)). For the reason behind this, please read the proofs in the original paper or a previous paper of [SQN(Soft Q Network/Soft Q Learning)](https://arxiv.org/abs/1702.08165). These two papers shares the same idea of bringing in entropy. The difference is that SAC minimizer the KL divergence while SQN directly sample actions proportional to exp(Q(s_t,*)/alpha), which is harder.

## Some tricks used
To maximize J(pi), the algorithm keeps track of V(s_t) = E_{a_t~pi}[Q(s_t,a_t)-log(pi(*|s_t))] and Q(s_t,a_t) = E[r_t+V(s_{t+1})]. As you can see mathematically V is not necessary but it is claimed to stablize the training. Two Q networks are updated separately for the same reason (double Q learning).

In practice of course we will use networks for Q, V and pi. The updates of Q, V networks are trivial but the update of pi requires a re[arametrization trich, which is not trivial. You can read the code for more details.

## Important Hyperparameters
In this implement, I have two hyperparameters that serve for the same purpose. scale and alpha. The rewards 'remembered' by the agent is real_reward*scale and alpha always accompanies the entropy term. Both these hyperparameters balances the importance of true reward and exploration. These two variables are the most sensitive hyperparameters to tune. I always set alpha=1 and tune scale.

## Others
I also tried different batchsize and different update details. Please read the code for more details. The effect of entropy term is tested by degenerate the algorithm back to DDPG but with two Q networks and a V network.
