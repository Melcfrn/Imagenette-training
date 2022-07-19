###############################################################################
# Packages                                                                    #
###############################################################################

from abc import ABC, abstractmethod

###############################################################################
# Data Loader                                                                 #
###############################################################################


class Loader(ABC):
    """
    This class implements a dataloader 
    """

    @abstractmethod
    def __init__(self, location, batch_size, shuffle):
        self.location = location
        self.batch_size = batch_size
        self.shuffle = shuffle

    @abstractmethod
    def get_dataset(self):
        
        pass