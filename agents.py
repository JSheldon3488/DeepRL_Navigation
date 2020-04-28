import numpy as np
import random
from collections import namedtuple, deque
from p1_navigation.neural_network import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

# Set up to run on GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
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
        # unpack experiences
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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize Replay Buffer Object for storing experience tuples

        :param action_size: how many action options there are
        :param buffer_size: how many experiences we can store at once
        :param batch_size: how many experiences per update
        :param seed: used for random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add experience to memory

        :param state: previous state
        :param action: action taken in previous state
        :param reward: reward for action taken
        :param next_state: state we ended up in after action is taken
        :param done: bool letting us know if we hit the terminal state
        :return: Void, just updates memory buffer
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory.

        :return: tensors for the randomly sampled states, actions, rewards, next_states, and dones.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # Note: np.vstack creates a column vector (vertical stack items in array)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)