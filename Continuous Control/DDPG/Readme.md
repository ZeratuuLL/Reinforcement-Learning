This is a failure. I tried a smaller learning rate, I tried to learn critic multiple steps with learn actor one step. I tried to clip the norm but nothing helps. Th absolutely value of my actor's output rom last hidden layer always goes larger. And thus a vanishing gradient in the actor. I checked, within 10 episodes the action' target network's output will be exactly the same as local network. I will keep trying to fix this problm but I don' give it much hope.

Here ar my attemps:
  * Do not initialize local and target network with same values.
