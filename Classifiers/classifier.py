###############################################################################
# Packages                                                                    #
###############################################################################

from abc import ABC, abstractmethod

###############################################################################
# Pytorch Classifier                                                          #
###############################################################################


class Classifier(ABC):
    """
    This class implements a neural network classifier
    with the PyTorch framework
    """

    @abstractmethod
    def __init__(self, model, criterion, optimizer, batch_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size

    @abstractmethod
    def fit(self, dataset, nb_epochs):
        """
        Train the classifier on the training set `dataset`.

        :param dataset: Training set.
        :param nb_epochs: Number of epochs to train.
        """

        pass

    @abstractmethod
    def predict(self, dataset):
        """
        Predict the `dataset` with the classifier.

        :param dataset: Testing set.
        """

        pass

    @abstractmethod
    def save(self, filename, path="Networks"):
        """
        Save a model to file in the format of Pytorch framework

        :param filename: Name of the file where is save the model.
        :param path: path where to save the file.
        """

        pass

    @abstractmethod
    def load(self, model, path_to_file):
        """
        Load a pytorch saved model (like .pt). You need to define your model in
        the networks_to_load.py file and call it in the function as var 'model'.

        :param model: Architecture of the model to load.
        :param path_to_file: path to the file where is the model to load.
        """
        pass
