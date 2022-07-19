###############################################################################
# Packages                                                                    #
###############################################################################

import torchvision
import torch
from Loaders.loader import Loader

###############################################################################
# Data Loader                                                                 #
###############################################################################


class PytorchLoader(Loader):
    """
    This class implements a dataloader for Pytorch models
    """

    def __init__(self, location, transform, batch_size, shuffle):
        self.location = location
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataset(self):
        set = torchvision.datasets.ImageFolder(
            root=self.location, transform=self.transform)
        loader = torch.utils.data.DataLoader(
            dataset=set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0)
        return loader
