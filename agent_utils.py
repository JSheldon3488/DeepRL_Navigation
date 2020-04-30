from collections import namedtuple, deque
import random
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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