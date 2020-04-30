import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from p1_navigation.neural_networks import QNetwork, Dueling_QNetwork
from p1_navigation.agent_utils import ReplayBuffer

# Set up to run on GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN_Agent():
    """Object that can interact with and learn from an environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object

        :param state_size: (int) dimension of each state
        :param action_size: (int) number of possible actions
        :param seed: random seed used for reproducible results
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Hyperparameters
        self.buffer_size = int(1e5)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.learning_rate = 0.0005
        self.update_rate = 4

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory setup
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def __str__(self):
        return "DQN_Agent"

    def step(self, state, action, reward, next_state, done):
        """Uses the information from the enviornment to update the local and target networks

        :param state: previous state we acted in
        :param action: previous action we took
        :param reward: reward for taking action from environment
        :param next_state: next state we need to act in
        :param done: bool representing if we came to a terminal state
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every update_rate time steps.
        self.t_step += 1
        if self.t_step%self.update_rate == 0:
            self.t_step = 0
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
            # ------------------- soft update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state, eps=0.):
        """Returns an action for a given state based on the current policy

        :param state: current state of the environment
        :param eps: hyperparameter for epsilon-greedy action selection
        :return: int representing an action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Turn off training and gradient calculations and just get the action_values
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()  # Turning back on training for the future

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """Update network parameters using given batch of experience tuples (using DQN)

        :param experiences: tuple of (s,a,r,s',done) tuples Tuple[torch.Variable])
        :param gamma: discount factor for update equation
        """
        # Unpack Experiences
        states, actions, rewards, next_states, dones = experiences

        # ---------------  DQN Update Method ---------------- #
        # Get max Q values for each of the next_states from target model
        next_Qvalues = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        td_target_values = rewards + (gamma * next_Qvalues * (1 - dones))
        # Get expected Q values from local model for all actions taken
        prev_qvalues = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(td_target_values, prev_qvalues)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """θ_target = τ*θ_local + (1 - τ)*θ_target
        target network moves in the direction of local network by TAU amount

        :param local_model: local neural network that is updated every time step
        :param target_model: target neural network that is decoupled from local network.
        :param tau: amount to move target network in the direction of local network
        :return: Void, just updates the target network weights
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DoubleDQN_Agent(DQN_Agent):
    """Double DQN Agent that interacts with and learns from the environment"""

    def __init__(self, state_size, action_size, seed):
        """Initialize a Double DQN Agent object using inheritance from DQN_Agent.

        :param state_size: (int) dimension of each state
        :param action_size: (int) number of possible actions
        :param seed: random seed used for reproducible results
        """
        super().__init__(state_size, action_size, seed)

    def __str__(self):
        return "Double_DQN_Agent"

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples. (Double DQN)

        :param experiences: tuple of (s,a,r,s',done) tuples Tuple[torch.Variable])
        :param gamma: discount factor for update equation
        """
        # Unpack Experiences
        states, actions, rewards, next_states, dones = experiences

        #-------------------- Double DQN Update -------------------#
        # Find max actions for next states based on the local_network
        local_argmax_actions = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
        # Use local_best_actions and target_network to get predicted value for next states
        next_qvalues = self.qnetwork_target(next_states).gather(1,local_argmax_actions).detach()

        #          Everything else same as DQN             #
        # Compute Q target values for current states (1-dones computes to 0 if next state is terminal)
        td_target_values = rewards + (gamma * next_qvalues * (1 - dones))
        # Get expected Q values from local model for all actions taken
        prev_qvalues = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(td_target_values, prev_qvalues)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Dueling_DDQN_Agent(DoubleDQN_Agent):
    "Dueling DQN Agent is the same as the Double DQN Agent except for the neural netowrks architecture"

    def __init__(self, state_size, action_size, seed):
        """Initialize a Dueling Double DQN Agent object using inheritance from Double DQN Agent.

        :param state_size: (int) dimension of each state
        :param action_size: (int) number of possible actions
        :param seed: random seed used for reproducible results
        """
        super().__init__(state_size, action_size, seed)
        # Use Dueling Network Architecture instead
        self.qnetwork_local = Dueling_QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = Dueling_QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)


    def __str__(self):
        return "Dueling_DDQN_Agent"