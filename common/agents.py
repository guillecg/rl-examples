from tqdm import tqdm

from collections import deque

from common.base import BaseAgent
from common.utils import epsilon_greedy_choice


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

        self.align_target_net()

        # RL parameters
        self.alpha = 0.01   # Learning rate
        self.gamma = 0.90   # Discount
        self.epsilon = 0.10 # Random action choice (epsilon greedy)

        self.exp_replay_len = 2000
        self.experience_replay = deque(maxlen=self.exp_replay_len)

        # Update attributes with kwargs
        self.__dict__.update(**kwargs)

    def align_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Avoid training on target network

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append(
            (state, action, reward, next_state, terminated)
        )

    def choose_action(self, state):
        # Change to (Batch, Channel, Width, Height)
        state = state.unsqueeze(0)

        # Flatten values to ease finding the max (forget about dimensions)
        q_values = self.policy_net(state).flatten()

        return epsilon_greedy_choice(values=q_values, epsilon=self.epsilon)

    def perform_train(self, n_timesteps=100, n_episodes=100):
        for episode in tqdm(range(1, n_episodes + 1)):
            for timestep in range(1, n_timesteps + 1):
                pass

    def perform_test(self, n_timesteps=20, n_episodes=20):
        for episode in tqdm(range(1, n_episodes + 1)):
            for timestep in range(1, n_timesteps + 1):
                pass
