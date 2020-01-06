import gym

import torch

from common.utils import image_to_pytorch


class GenericEnvWrapper(gym.Wrapper):
    ''' Helper wrapper for getting the number of actions in generic environments
    '''

    def __init__(self, env):
        super(GenericEnvWrapper, self).__init__(env)

        self.available_states = env.observation_space
        self.available_actions = env.action_space

    @property
    def n_actions(self):
        return len(self.available_actions)


class RoboticsEnvWrapper(gym.Wrapper):
    ''' Helper wrapper for getting the number of actions in robotic environments
    '''

    def __init__(self, env):
        super(RoboticsEnvWrapper, self).__init__(env)

        self.available_states = env.observation_space.spaces.keys()
        self.available_actions = env.action_space

    @property
    def n_actions(self):
        return self.available_actions.shape[0]


class ImageEnvWrapper(gym.Wrapper):
    ''' Helper wrapper for returning images in different environments
    Adapted from: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html#using-custom-environments
    '''
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, env, img_width=84, img_height=84):
        super(ImageEnvWrapper, self).__init__(env)

        self.base_env = env
        self.img_width  = img_width
        self.img_height = img_height

    def render(self, mode='rgb_array'):
        return self.base_env.render(
            mode,
            width=self.img_width,
            height=self.img_height
        )


class ScaledImageEnvWrapper(gym.Wrapper):
    ''' Helper wrapper for returning scaled images
    '''

    def __init__(self, env, width=84, height=84, scaling_fn=None):
        super(ScaledImageEnvWrapper, self).__init__(env)

        self.base_env = env
        self.width  = width
        self.height = height

        self.scaling_fn = scaling_fn if scaling_fn else self.normalization

    def render(self, mode='rgb_array'):
        env_img = self.base_env.render(
            mode,
            width=self.width,
            height=self.height
        )

        return env_img

    @staticmethod
    def normalization(img):
        return img / 255.0


class PyTorchScaledImageEnvWrapper(gym.Wrapper):
    ''' Helper wrapper for returning scaled images as Torch tensors
    '''

    def __init__(self, env, width=84, height=84, scaling_fn=None, cuda=True):
        super(PyTorchScaledImageEnvWrapper, self).__init__(env)

        self.base_env = env
        self.width  = width
        self.height = height

        self.cuda = cuda
        self.scaling_fn = scaling_fn if scaling_fn else self.normalization

    def render(self, mode='rgb_array'):
        env_img = self.base_env.render(
            mode,
            width=self.width,
            height=self.height
        )

        env_img = self.scaling_fn(env_img)
        env_img = image_to_pytorch(img=env_img, cuda=self.cuda)

        return env_img

    @staticmethod
    def normalization(img):
        return img / 255.0
