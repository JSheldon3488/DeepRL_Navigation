# Udacity Deep Reinforcement Learning Project 1: Navigation
Starter code and project details can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

## Agents
This section covers all the RL Agents that were implemented to solve goal for this environment. Each section will have an explanation of the agent and neural network used by that agent,
 a graph showing the results of that agents learning in the environment, and references to the papers relevant to that agent.
 
 ### Deep Q-Network
 This agent is a Deep Q-Netork Agent similar to the one designed in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). 
 This agent uses Experience Replay, and Fixed Q-Targets (both are explained in the paper). Details about how these concepts are being implemented can be seen in [agents.py](agents.py).
 Experience Replay is accomplished by using the ReplayBuffer class and the batch sampling/learning is done in Agent.step.
 Fixed Q-Targets is accomplished by using a local network and target network and updating the target network parameters every so often in the direction of the local network parameters using Agent.soft_update.
 
 **Neural Network Architecture:**
 Four fully connected layers with relu activation functions. Using Adam optimizer with a learning rate of 0.0005.
 * Layer1: State_Size(37), 128
 * Layer2: 128, 64
 * Layer3: 64, 32
 * Layer4: 32, action_size(4)
 
 **Hyperparameters:** 
* Replay Buffer Size: 100,000
* Batch Size: 64
* Discount Rate: 0.99 (Q-Value Calculations)
* Network Update Rate (Tau): 0.001
* Learning Rate: 0.0005 (Neural Network)
* Update Rate: 4 (how often to update networks)

**Results:**
<p align="center">
    <img src="/images/DQN_Agent_.png">
</p>
 
 ### Double Deep Q-Network
 
 ## Dueling Deep Q-Network
 
 ## Comparisons
 This section compares the RL Agents.