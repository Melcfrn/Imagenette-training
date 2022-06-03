###############################################################################
# Packages                                                                    #
###############################################################################

import torchvision
import torch
import multiprocessing

###############################################################################
# Data Loader                                                                 #
###############################################################################


class Loader():
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
            num_workers=multiprocessing.cpu_count()-1)
        return loader
