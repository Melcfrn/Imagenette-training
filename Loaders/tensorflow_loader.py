###############################################################################
# Packages                                                                    #
###############################################################################

from msilib import Directory
import tensorflow as tf
from tensorflow import keras
from Loaders.loader import Loader

###############################################################################
# Data Loader                                                                 #
###############################################################################


class TensorflowLoader(Loader):
    """
    This class implements a dataloader for Tensorflow models
    """

    def __init__(self, location, size, batch_size, shuffle, dataaug):
        self.location = location
        self.size = size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataaug=dataaug

    def get_dataset(self):
        loader = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.location,
            label_mode='categorical',
            image_size=(self.size, self.size),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        loader = loader.map(lambda x, y: (self.dataaug(x, training=True), y))

        loader = loader.prefetch(buffer_size=self.size)
        return loader