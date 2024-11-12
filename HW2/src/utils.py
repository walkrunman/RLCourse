import random
from collections import namedtuple, deque

import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 
                                       'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity, device):
        self.memory = deque([], maxlen=capacity)
        self.device = device

    def push(self, *args):
        """Save a transition"""
        state, action, next_state, reward, done_ = args

        if next_state is None:
            next_state = torch.Tensor([[0 for x in range(8)]]).to(self.device)

        self.memory.append((state, action, next_state, reward, done_))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

    
