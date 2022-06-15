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
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(128 * 38 * 38, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.3),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.network(x)
