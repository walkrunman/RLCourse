import numpy as np

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(np.prod(observation_space.shape), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, np.prod(action_space.shape))

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2, dtype=torch.float32
            ),
        )

    def forward(self, state):
        state = nn.ReLU()(self.fc1(state))
        state = nn.ReLU()(self.fc2(state))
        action = nn.Tanh()(self.fc3(state))

        action = action * self.action_scale + self.action_bias
        return action


class Critic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(
            np.prod(observation_space.shape) + np.prod(action_space.shape), hidden_dim
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q = torch.cat([state, action], dim=1)
        q = nn.ReLU()(self.fc1(q))
        q = nn.ReLU()(self.fc2(q))
        q = self.fc3(q)
        return q
