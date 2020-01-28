SEED = 42

import numpy as np
np.random.seed(SEED)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from tqdm import tqdm

from common.base import BaseAgent
from common.utils import epsilon_greedy_choice, ReplayMemory


class DQN(BaseAgent):

    def __init__(self, env, policy, device, input_shape=None, **kwargs):
        super(DQN, self).__init__(env)

        self.device = device
        self.input_shape = (1, 500, 500) if not input_shape else input_shape

        # Set policy and target networks
        self.policy_net = policy(
            input_shape=self.input_shape,
            n_actions=self.env.n_actions
        )

        self.target_net = policy(
            input_shape=self.input_shape,
            n_actions=self.env.n_actions
        )

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self._align_target_net()

        # RL parameters
        self.alpha = 0.01   # Learning rate
        self.gamma = 0.90   # Discount
        self.epsilon = 0.10 # Random action choice (epsilon greedy)

        # Training parameters
        self.batch_size = 32
        self.loss_fn = F.smooth_l1_loss # Huber loss
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.exp_replay_len = 2000
        self.memory = ReplayMemory(max_size=self.exp_replay_len)

        # Update attributes with kwargs
        self.__dict__.update(**kwargs)

    def _align_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Avoid training on target network

    def choose_action(self, state):
        # Flatten values to ease finding the max (forget about dimensions)
        q_values = self.policy_net(state).flatten()

        return epsilon_greedy_choice(values=q_values, epsilon=self.epsilon)

    def _store(self, state, action, reward, next_state, terminated):
        self.memory.append(
            (state, action, reward, next_state, terminated)
        )

    def _retrain(self):
        minibatch = self.memory.sample(batch_size=self.batch_size)

        for state, action, reward, next_state, terminated in minibatch:
            observed_values = self.policy_net(state).flatten()
            target_values = self.target_net(next_state).flatten()

            # Update target values as the discounted reward
            target_values = reward + self.gamma * target_values

            # Calculate loss
            loss = self.loss_fn(target_values, observed_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()

            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()

    def perform_train(self, n_timesteps=100, n_episodes=100):
        for episode in tqdm(range(1, n_episodes + 1)):
            # Reset at the beginning of each episode
            self.env.reset()
            reward = 0

            # Initialize state in each episode
            state = self.env.render()

            # Change to (Batch, Channel, Width, Height)
            state = state.unsqueeze(0)

            for timestep in range(1, n_timesteps + 1):
                # Choose action for the current state
                action = self.choose_action(state)

                # Perform chosen action
                _, reward, terminated, info = self.env.step(action)

                # Render state as RGB image
                next_state = self.env.render()

                # Change to (Batch, Channel, Width, Height)
                next_state = next_state.unsqueeze(0)

                # Store in the experience replay memory/buffer
                self._store(state, action, reward, next_state, terminated)

                # Finish iteration by replacing state as the new state
                # Note: copy and detach the tensor from the computation graph
                state = next_state.clone().detach()

                if timestep > self.batch_size:
                    self._retrain()
                    self._align_target_net()

                if terminated:
                    self._align_target_net()


    def perform_test(self, n_timesteps=20, n_episodes=20):
        for episode in tqdm(range(1, n_episodes + 1)):
            # Reset at the beginning of each episode
            self.env.reset()
            reward = 0

            # Initialize state in each episode
            state = self.env.render()

            # Change to (Batch, Channel, Width, Height)
            state = state.unsqueeze(0)

            for timestep in range(1, n_timesteps + 1):

                state_array = state.cpu().numpy()[0]
                plt.imshow(state_array[0])
                plt.show()

                # Choose action for the current state
                action = self.choose_action(state)

                # Perform chosen action
                _, reward, terminated, info = self.env.step(action)

                # Render state as RGB image
                next_state = self.env.render()

                # Change to (Batch, Channel, Width, Height)
                next_state = next_state.unsqueeze(0)

                # Finish iteration by replacing state as the new state
                # Note: copy and detach the tensor from the computation graph
                state = next_state.clone().detach()
