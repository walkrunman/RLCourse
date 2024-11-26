import torch


class OrnsteinUhlenbeckNoise:
    def __init__(self, size: int, mu=0.0, sigma=0.1, theta=0.15):
        self.mu = mu * torch.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.size = size
        self.state = self.mu.clone()

    def __call__(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * torch.randn(
            self.size
        )
        return self.state.clone()

    def reset(self):
        self.state = self.mu.clone()
