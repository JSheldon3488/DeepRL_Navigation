import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Dueling_QNetwork(nn.Module):
    """ Dueling Neural Network architecture with state value and action advantage evaluation """

    def __init__(self, state_size, action_size, seed):
        """ Initialize parameters and set up layers of the model

        :param state_size: Dimension of state space
        :param action_size: Dimension of action space
        :param seed: Random seed
        """
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manager_seed(seed)
        # Shared Layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Individual Layers
        self.fc_value = nn.Linear(64,32)
        self.fc_action = nn.Linear(64,32)
        self.state_value = nn.Linear(32,1)
        self.action_advantage = nn.Linear(32,action_size)

    def forward(self, state):
        """ Dueling network that maps state to action values using a Dueling network architecture"""
        # Shared Layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Individual Layers
        state_value = F.relu(self.fc_value(x))
        action_advantage = F.relu(self.fc_action(x))
        state_value = self.state_value(state_value)
        action_advantage = self.action_advantage(action_advantage)

        # Combine for final value using equation (9) from paper
        return state_value + (action_advantage - action_advantage.mean())