import torch
import torch.nn as nn
from DefModels.model import Model, ModelTF

def cifar_conv() -> nn.Sequential:
        return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(32),
                nn.Dropout(0.3),

                nn.Conv2d(32, 64, kernel_size=5, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.Dropout(0.3),

                nn.Conv2d(64, 128, kernel_size=5, padding=1),
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

def imagenette_conv():
        return Model(3, [32, 64, 128], [512], 10, activation='relu')