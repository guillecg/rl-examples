import gym

# Needed for rendering in RGB, see:
# https://github.com/openai/mujoco-py/issues/390
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

from common.policies import CNNPolicy
from common.utils import image_to_pytorch


ENV_NAME = 'FetchReach-v1'


if __name__ == '__main__':
    env = gym.make(ENV_NAME)

    # Get the input shape by rendering the enviroment
    env_img = env.render(mode='rgb_array')

    # Reshape to (channels, height, width) to fit PyTorch dimensions:
    # from (500, 500, 3) to (3, 500, 500)
    env_img = image_to_pytorch(env_img)

    # Build the model
    policy = CNNPolicy(input_shape=env_img.shape)
