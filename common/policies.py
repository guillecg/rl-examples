# Reproducibility
SEED = 42

import torch
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# TODO: add arquitecture from ONNX, JSON or similar format files

import numpy as np

import torch.nn as nn

from collections import OrderedDict

from common.utils import get_network_output_shape


class CnnPolicy(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(CnnPolicy, self).__init__()

        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=2)),
            ('conv1-batch_norm', nn.BatchNorm2d(num_features=32, affine=True)),
            ('conv1-relu', nn.ReLU()),
            # ('conv1-max_pool', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv2', nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=5)),
            ('conv2-batch_norm', nn.BatchNorm2d(num_features=64, affine=True)),
            ('conv2-relu', nn.ReLU()),
            # ('conv2-max_pool', nn.MaxPool2d(kernel_size=3, stride=2)),

            # ('conv3', nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=5)),
            # # ('conv3-batch_norm', nn.BatchNorm2d(32)),
            # ('conv3-relu', nn.ReLU()),
            # # ('conv3-max_pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        cnn_output_shape = get_network_output_shape(
            network=self.cnn,
            input_shape=input_shape
        )

        self.dnn = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(np.prod(cnn_output_shape), 128)),
            ('dense1-relu', nn.ReLU()),
            ('dense2', nn.Linear(128, 32)),
            ('dense2-relu', nn.ReLU()),
            ('dense3', nn.Linear(32, n_actions)),
            ('dense3-softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, x):
        hidden = self.cnn(x)

        # Flattens CNN output, see: https://stackoverflow.com/a/49607525
        hidden = hidden.view(hidden.shape[0], -1)

        output = self.dnn(hidden)

        return output
