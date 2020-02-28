import gym

# Needed for rendering in RGB, see:
# https://github.com/openai/mujoco-py/issues/390
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

import torch

from modules.wrappers import (
    RoboticsEnvWrapper,
    PyTorchScaledImageEnvWrapper
)
from modules.policies.cnn import CnnPolicy
from modules.agents.dqn import DQN


ENV_NAME = 'FetchReach-v1'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # NOTE: use dense reward type to fully capture the distance between
    # the gripper and the goal, otherwise the enviroment return a sparse
    # reward of -1.0 always unless it is 0.05 (units?) from the goal
    env = gym.make(ENV_NAME, reward_type='sparse')

    # WARNING: not working
    # Avoid the presence of target in the air (3D solution not available yet)
    # https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/reach.py#L10

    # Custom wrappers allow returning an image using render
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
    agent.perform_train(n_timesteps=200, n_episodes=100)
    agent.perform_test(n_timesteps=50, n_episodes=2)
