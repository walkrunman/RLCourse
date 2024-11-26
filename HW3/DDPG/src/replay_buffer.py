import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        observation, action, reward, next_observation, terminated, truncated = (
            transition
        )
        self.buffer.append((observation, action, reward, next_observation, terminated))

    def sample(self, batch_size):
        observations, actions, rewards, next_observations, terminations = zip(
            *random.sample(self.buffer, batch_size)
        )
        return observations, actions, rewards, next_observations, terminations

    def __len__(self):
        return len(self.buffer)
