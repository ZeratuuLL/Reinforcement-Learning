## Document:
  * **infrasturcture.py**: where I saves replaybuffer and networks
  * **Policy_Agents.py**: where I saves the agent (now only DDPG)
  * **Reacher.py**: Run this in the terminal to start training
  * **algorithms**: implements of algorithms (now only DDPG)
  
## History
This is a failure. I tried a smaller learning rate, I tried to learn critic multiple steps with learn actor one step. I tried to clip the norm but nothing helps. Th absolute value of my actor's output of last hidden layer always goes larger. And thus a vanishing gradient in the actor. I checked, within 10 episodes the action' target network's output will be exactly the same as local network. I will keep trying to fix this problm but I don' give it much hope.

Here are my attemps:
  * Move the position of action in critic network from second hidden layer to first hidden layer. This should make the gradient smaller (in absolute value). **This frees actor to give actions on the margin. But the learning is still very slow. I noticed that the local and target critic gives same estimates at the very end. Will see if this is true for all**
  
  * This is true. Within 5 episodes the values of two critics has error smaller than 1%. The same is true for two actors within 10 episodes. What's more, the critics are still learning but the actors are almost resting. **Now I will try a larger learning rate (0.0001 --> 0.01).**
  
  * I don't think that works. The two critics get close (within 1% error) within 10 episodes. And the learning is ruined. So I think this is not a good idea. But the actors is always learning, which is great! What's more, I found that the critics are actually both learning but close to each other. But sometimes they give negative estimates. **Learning rate set back (0.01 --> 0.0001) an set tau (soft update parameter) smaller (0.001 --> 0.0001)**
  
  * This destroy the learning process... Turning the tau back **(0.0001 --> 0.001)** It's learning again. BTW **(lr=tau=0.001)** is unstable, I have seen it succeeded as well as failed. This time it seems good. The local and target networks are somehow different not. **Next: force the critic target network to output none-negative values**
  
  * This makes the critics predict positive values within 5 episodes, which is good. The actors are still going to the margins, but much slower. Not the problems should be with the critics. I noticed that the prediction values of both critics are always increasing. Is that increasing too slow that my agent fails to learn? **Next: do not initialize local and target networks with same weights.**
  
  * This is a bad idea. **Return to the original version and train with a longer period (50 --> 200).**
  
  * Yes, a larger number of training episodes did bring me an improvement. At around 70 episodes the performance boosted a lot and get stablized. I noticed that the critic network is still learning. Perhaps we still need an even longer time to train. **Perhaps improving the critics multiple times will also help. But I will not try this at present. Next: 2000 episodes training.**
  
  * I do not expect much more from this. I see the critics' predictions no longer increasing at aroung 500 episodes. I am thinking for the following:
    * Making replaybuffer larger. Now it's 1e5, and each episode we accumulate 20000 experiences. So we are only remembering past 5 episodes. I will make this number to be 1e6 so we can remember 50 episodes.
    * Try to use different samples to update actors and critics. If this doesn't hurt we can try to update actors/critics with different times and different samples. At the same time, we can choose to learn after several timesteps. For example, learn actor 3 times, critics 5 times after every 5 timesteps.
    * **Next: Use different samples**
  
  * I think that somehow hurts. Not a surprise. **Next: set replaybuffer larger**
  
  * **Next: set learning rate for actor smaller(0.1 if critic --> 0.01 of critic)**
