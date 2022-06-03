###############################################################################
# Packages                                                                    #
###############################################################################

import torch.nn as nn

###############################################################################
# Model                                                                       #
###############################################################################


class Model(nn.Module):
    """
    This class implements a neural network from PyTorch framework
    """

    def __init__(self):
        super().__init__()
        self.kernel_size = 5
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 10)
        )

    def forward(self, x):
        return self.network(x)
