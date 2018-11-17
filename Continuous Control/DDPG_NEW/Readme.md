Here I will restart the DDPG algorithm. Nothing fancy. Nothing flexble. Just the most basic type especially for this question.

Files:
  * **Continuous_Control-Copy2.ipynb**: A jupyter notebook which can be used to train the agent locally. It also contains the testing part to show that my code works correctly. This file is then splited into the following three files in order to run on a server. I know that some outputs are too long. But I cannot fold them here. I apologize for that....
  * **infrastructures.py**: Where I put actor network, critic network, replaybuffer and OUNoise
  * **agents.py**: Where I put the agent for DDPG algorithm
  * **DDPG.py**: The file that can be called on a server to process the training.
  
If you find any bugs, please kindly let me know. I would really appreciate that!
