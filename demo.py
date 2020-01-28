import gym

# Needed for rendering in RGB, see:
# https://github.com/openai/mujoco-py/issues/390
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

import torch

from common.wrappers import (
    RoboticsEnvWrapper,
    PyTorchScaledImageEnvWrapper
)
from common.policies import CnnPolicy
from common.agents import DQN


ENV_NAME = 'FetchReach-v1'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # Custom wrappers allow returning an image using render
    env = gym.make(ENV_NAME)
    env = RoboticsEnvWrapper(env)
    env = PyTorchScaledImageEnvWrapper(env=env, width=150, height=150)

    # Get the input shape for the policy network by rendering the enviroment
    input_shape = env.render().shape

    # Build the agent using a policy network
    agent = DQN(
        env=env,
        policy=CnnPolicy,
        device=DEVICE,
        input_shape=input_shape
    )
    agent.perform_train(n_timesteps=100, n_episodes=100)
    agent.perform_test(n_timesteps=10, n_episodes=2)
