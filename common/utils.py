# Reproducibility
SEED = 42

import random
random.seed(SEED)

import torch
torch.manual_seed(SEED)

from copy import copy

from collections import namedtuple


def epsilon_greedy_choice(values, epsilon=0.05):
    ''' Helper function for choosing the action using epsilon greedy
    '''
    chosen_action = None

    if torch.rand(1) <= epsilon:
        chosen_action = torch.randint(low=0, high=4, size=(1,))[0]
    else:
        chosen_action = torch.argmax(values)

    # Convert to array to avoid further errors
    chosen_action = chosen_action.cpu().numpy()

    # Convert to integer
    chosen_action = int(chosen_action)

    return chosen_action


def image_to_pytorch(img, cuda=True):
    ''' Helper function for converting the shape to PyTorch's compatible:
        - Numpy's order: (Width, Height, Channel)
        - Torch's order: (Channel, Width, Height)
    '''
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)

    if cuda:
        img = img.type(torch.cuda.FloatTensor)

    return img


def get_network_output_shape(network, input_shape):
    ''' Helper function to calculate output size in PyTorch
    '''
    network_output = copy(network)(torch.zeros(1, *input_shape))

    return network_output.shape


class ReplayMemory:
    ''' Extracted from: https://stackoverflow.com/a/56171119
    '''

    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]
