Here I will restart the DDPG algorithm. Nothing fancy. Nothing flexble. Just the most basic type especially for this question.

Files:
  * **Continuous_Control-Copy2.ipynb**: A jupyter notebook which can be used to train the agent locally. It also contains the testing part to show that my code works correctly. This file is then splited into the following three files in order to run on a server. I know that some outputs are too long. But I cannot fold them here. I apologize for that....
  * **infrastructures.py**: Where I put actor network, critic network, replaybuffer and OUNoise
  * **agents.py**: Where I put the agent for DDPG algorithm
  * **DDPG.py**: The file that can be called on a server to process the training.
  
If you find any bugs, please kindly let me know. I would really appreciate that!


## History:

  * Now I have added batch normalization to the activation of first hidden layer. This works great. The hyper parameters are: 
    * batch size: 128
    * learning rate for actor: 0.0001
    * learning rate for critic: 0.001
    * soft update parameter: 0.001
    * replaybuffer size: 1e6
    * $\gamma$: 0.99
   
  * Instead of update networks once every timestep, I tried to update them 10 times every 20 timestep. It's still learning after 1000 episodes. Trying 2000.
    
