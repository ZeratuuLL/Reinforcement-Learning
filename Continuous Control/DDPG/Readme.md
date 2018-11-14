## Document:
  * **infrasturcture.py**: where I saves replaybuffer and networks
  * **Policy_Agents.py**: where I saves the agent (now only DDPG)
  * **Reacher.py**: Run this in the terminal to start training
  * **algorithms**: implements of algorithms (now only DDPG)
  
## History
This is a failure. I tried a smaller learning rate, I tried to learn critic multiple steps with learn actor one step. I tried to clip the norm but nothing helps. Th absolutely value of my actor's output rom last hidden layer always goes larger. And thus a vanishing gradient in the actor. I checked, within 10 episodes the action' target network's output will be exactly the same as local network. I will keep trying to fix this problm but I don' give it much hope.

Here ar my attemps:
  * Move the position of action in critic network from second hidden layer to first hidden layer. This should make the gradient smaller (in absolute value). **This frees actor to give actions on the margin. But the learning is still very slow. I noticed that the local and target critic gives same estimates at the very end. Will see if this is true for all**
  
  * This is true. Within 5 episodes the values of two critics has error smaller than 1%. The same is true for two actors within 10 episodes. What's more, the critics are still learning but the actors are almost resting. **Now I will try a larger learning rate (0.0001 --> 0.01).**
  
  * I don't think that works. The two critics get close (within 1% error) within 10 episodes. And the learning is ruined. So I think this is not a good idea. But the actors is always learning, which is great! What's more, I found that the critics are actually both learning but close to each other. But sometimes they give negative estimates. **Learning rate set back (0.01 --> 0.0001) an set tau (soft update parameter) smaller (0.001 --> 0.0001)**
  
  * This destroy the learning process... Turning the tau back **(0.0001 --> 0.001)** It's learning again. BTW **(lr=tau=0.001)** is unstable, I have seen it succeeded as well as failed. **Next: do not initialize local and target networks with same weights.**
