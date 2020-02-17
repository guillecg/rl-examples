import gym
from gym.envs.robotics.fetch_env import goal_distance

import numpy as np

import cv2

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

    def __init__(self, env, action_mapping={}):
        super(RoboticsEnvWrapper, self).__init__(env)

        self.available_states = env.observation_space.spaces.keys()
        self.available_actions = env.action_space

        # Set action mapping: number to binary array
        self.action_mapping = action_mapping or {
            0: [+1., 0., 0., 0.],    # Go: forward
            1: [-1., 0., 0., 0.],    # Go: backwards
            2: [0., +1., 0., 0.],    # Go: right
            3: [0., -1., 0., 0.],    # Go: left
            4: [0., 0., +1., 0.],    # Go: up
            5: [0., 0., -1., 0.],    # Go: down
            6: [0., 0., 0., +1.],    # Gripper: open
            7: [0., 0., 0., -1.],    # Gripper: close
            8: [0., 0., 0., 0.]      # Idle
        }

    @property
    def n_actions(self):
        return len(self.action_mapping.keys())


class ImageEnvWrapper(gym.Wrapper):
    ''' Helper wrapper for returning images in different environments
    Adapted from: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html#using-custom-environments
    '''
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, env, img_width=84, img_height=84):
        super(ImageEnvWrapper, self).__init__(env)

        self.img_width  = img_width
        self.img_height = img_height

    def render(self, mode='rgb_array'):
        return self.env.render(
            mode,
            width=self.img_width,
            height=self.img_height
        )

    def step(self, action):
        _, reward, done, info = self.env.step(action)

        return self.render(), reward, done, info


class ScaledImageEnvWrapper(gym.Wrapper):
    ''' Helper wrapper for returning scaled images
    '''
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, env, width=84, height=84, scaling_fn=None):
        super(ScaledImageEnvWrapper, self).__init__(env)

        self.width  = width
        self.height = height

        self.scaling_fn = scaling_fn or self.normalization

    def render(self, mode='rgb_array'):
        env_img = self.env.render(
            mode,
            width=self.width,
            height=self.height
        )

        return self.scaling_fn(env_img)

    @staticmethod
    def normalization(img):
        return img / 255.0

    def step(self, action):
        _, reward, done, info = self.env.step(action)

        return self.render(), reward, done, info


class PyTorchScaledImageEnvWrapper(gym.Wrapper):
    ''' Helper wrapper for returning scaled images as Torch tensors
    '''
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, env, width=84, height=84, scaling_fn=None, cuda=True):
        super(PyTorchScaledImageEnvWrapper, self).__init__(env)

        self.width  = width
        self.height = height

        self.cuda = cuda
        self.scaling_fn = scaling_fn or self.normalization

    def render(self, mode='rgb_array'):
        env_img = self.env.render(
            mode,
            width=self.width,
            height=self.height
        )

        env_img = self.extract_red(img=env_img)
        # env_img = self.apply_threshold_to_img(img=env_img)
        env_img = self.scaling_fn(env_img)
        env_img = image_to_pytorch(img=env_img, cuda=self.cuda)

        return env_img

    @staticmethod
    def extract_red(
            img,
            lower_red=np.array([0, 100, 100]),
            upper_red=np.array([10, 255, 255])
        ):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, lower_red, upper_red)

        mask = mask.reshape(*mask.shape, 1)

        return mask

    @staticmethod
    def apply_threshold(img, thr=200):
        _, threshold = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

        return threshold

    @staticmethod
    def normalization(img):
        return img / 255.0

    def step(self, action):
        _, reward, done, info = self.env.step(action)

        return self.render(), reward, done, info
