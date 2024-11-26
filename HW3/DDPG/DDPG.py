import os
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym

from src.models import Actor, Critic
from src.noise import OrnsteinUhlenbeckNoise
from src.replay_buffer import ReplayBuffer
from src.logger import Logger


class DDPG:
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make(self.config.env_name)
        observation_space, action_space = (
            self.env.observation_space,
            self.env.action_space,
        )

        self.actor = Actor(observation_space, action_space, self.config.hidden_dim).to(
            self.device
        )
        self.target_actor = Actor(
            observation_space, action_space, self.config.hidden_dim
        ).to(self.device)
        self.soft_update(self.actor, self.target_actor, 1.0)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.lr
        )

        self.critic = Critic(
            observation_space, action_space, self.config.hidden_dim
        ).to(self.device)
        self.target_critic = Critic(
            observation_space, action_space, self.config.hidden_dim
        ).to(self.device)
        self.soft_update(self.critic, self.target_critic, 1.0)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.lr
        )

        self.buffer = ReplayBuffer(self.config.buffer_capacity)
        self.noise = OrnsteinUhlenbeckNoise(
            size=np.prod(action_space.shape),
            mu=0.0,
            sigma=self.config.noise_sigma,
            theta=self.config.noise_theta,
        )

        self.logger = Logger(self.config.total_steps, self.config.num_checkpoints)

    def soft_update(self, online, target, tau):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def checkpoint(self, steps):
        if not os.path.exists("models"):
            os.makedirs("models")
        checkpoint_path = f"models/{self.config.env_name}_{steps}.pth"
        torch.save(self.actor.state_dict(), checkpoint_path)

    def select_action(self, observation, add_noise=False):
        with torch.no_grad():
            observation_tensor = torch.tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = self.actor(observation_tensor).squeeze(0)
            if add_noise:
                noise = self.actor.action_scale * self.noise().to(self.device)
                action = torch.clamp(
                    action + noise,
                    min=-self.actor.action_scale,
                    max=self.actor.action_scale,
                )
            return action.cpu().numpy()

    def learn(self):
        observations, actions, rewards, next_observations, terminations = (
            self.buffer.sample(self.config.batch_size)
        )

        observations = torch.tensor(
            np.array(observations), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.array(actions), dtype=torch.float32, device=self.device
        ).view(self.config.batch_size, -1)
        rewards = torch.tensor(
            np.array(rewards), dtype=torch.float32, device=self.device
        ).view(self.config.batch_size, 1)
        next_observations = torch.tensor(
            np.array(next_observations), dtype=torch.float32, device=self.device
        )
        terminations = torch.tensor(
            np.array(terminations), dtype=torch.float32, device=self.device
        ).view(self.config.batch_size, 1)

        # Critic loss and param update
        with torch.no_grad():
            next_state_q = self.target_critic(
                next_observations, self.target_actor(next_observations)
            )
            target_q = rewards + self.config.gamma * (1.0 - terminations) * next_state_q
        current_action_q = self.critic(observations, actions)
        critic_loss = F.mse_loss(current_action_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.grad_norm_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_norm_clip)
        self.critic_optimizer.step()

        # Actor loss and param update
        current_action_q = self.critic(observations, self.actor(observations))
        actor_loss = -(current_action_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.grad_norm_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_norm_clip)
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.actor, self.target_actor, self.config.tau)
        self.soft_update(self.critic, self.target_critic, self.config.tau)

    def train(self):
        observation, _ = self.env.reset()

        for step in range(1, self.config.total_steps + 1):
            if step > self.config.learning_starts:
                action = self.select_action(observation, add_noise=True)
            else:
                action = self.env.action_space.sample()

            next_observation, reward, terminated, truncated, _ = self.env.step(action)

            self.logger.log(reward, terminated, truncated)

            self.buffer.add(
                (observation, action, reward, next_observation, terminated, truncated)
            )

            if (
                len(self.buffer) > self.config.batch_size
                and step >= self.config.learning_starts
            ):
                self.learn()

            if terminated or truncated:
                next_observation, _ = self.env.reset()
                self.noise.reset()
            observation = next_observation

            # Print training info if verbose
            if self.config.verbose:
                self.logger.print_logs()

            # Save weights if checkpointing
            if self.config.checkpoint and step % self.logger.checkpoint_interval == 0:
                self.checkpoint(step)

            # Check stopping condition
            if (
                self.config.target_reward is not None
                and len(self.logger.episode_returns) >= 20
            ):
                mean_reward = np.mean(self.logger.episode_returns[-20:])
                if mean_reward >= self.config.target_reward:
                    if self.config.verbose:
                        print("\nTarget reward achieved. Training stopped.")
                    break