# Reproducibility
SEED = 42

import torch
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# TODO: add arquitecture from ONNX, JSON or similar format files

import numpy as np

import torch.nn as nn

from common.utils import get_network_output_shape


class CnnPolicy(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(CnnPolicy, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        cnn_output_shape = get_network_output_shape(
            network=self.cnn,
            input_shape=input_shape
        )

        self.dnn = nn.Sequential(
            nn.Linear(np.prod(cnn_output_shape), 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        hidden = self.cnn(x)

        # Flattens CNN output, see: https://stackoverflow.com/a/49607525
        hidden = hidden.view(hidden.shape[0], -1)

        output = self.dnn(hidden)

        return output
