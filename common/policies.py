from abc import abstractmethod

import numpy as np

import torch.nn as nn


# TODO: add arquitecture from ONNX, JSON or similar format files


class BaseDeepPolicy(nn.Module):

    def __init__(self, input_shape):
        super(BaseDeepPolicy, self).__init__()

        self.input_shape = input_shape

    @abstractmethod
    def build_network(self):
        ''' Abstract method to be replaced by the desired network architecture
        '''
        pass

    @abstractmethod
    def forward(self, x):
        ''' Abstract method to be replaced by the desired forward function
        '''
        pass

    @staticmethod
    def get_network_output_size(network, input_shape):
        network_output = network(torch.zeros(1, *input_shape))
        return network_output.flatten().shape[0]

    @property
    def network(self):
        return self.build_network()

    @property
    def network_output_size(self):
        return self.get_network_output_size(self.network, self.input_shape)


class CNNPolicy(BaseDeepPolicy):

    def build_network(self):
        return nn.Sequential(
            nn.Conv2d(self.input_shape[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)
