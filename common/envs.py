# Adapted from: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html#using-custom-environments

import gym
from gym import spaces


class MujocoImgEnv(gym.GoalEnv):
    ''' Custom environment for returning images in Mujoco environments
    '''
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, env):
        super(MujocoImgEnv, self).__init__()

        self.base_env = env
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

    def step(self, action):
        observation, reward, done, info = self.base_env.step(action)
        return observation, reward, done, info

    def reset(self):
        self.base_env.reset()
        return observation  # reward, done, info can't be included

    def render(self, mode='rgb_array'):
        return self.base_env.render(mode, width=500, height=500)

    def close(self):
        self.base_env.close()


if __name__ == '__main__':
    # Needed for rendering in RGB, see:
    # https://github.com/openai/mujoco-py/issues/390
    from mujoco_py import GlfwContext
    GlfwContext(offscreen=True)

    env = gym.make('FetchReach-v1')
    env = MujocoImgEnv(env)

    print('[+] Image returned from render:', env.render())
