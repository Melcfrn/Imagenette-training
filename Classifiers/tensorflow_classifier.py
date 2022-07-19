###############################################################################
# Packages                                                                    #
###############################################################################

import torch
import os
from time import sleep
from tqdm import tqdm
from Classifiers.classifier import Classifier

###############################################################################
# Pytorch Classifier                                                          #
###############################################################################


class TensorflowClassifier(Classifier):
    """
    This class implements a neural network classifier
    with the Tensorflow/Keras framework
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        batch_size=64
                 ):
        # TODO: Vérifier comment utiliser cpu/gpu dans tensorflow et implémenter dans tensorflowclassifier et dans main
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size

    def fit(self, dataset, nb_epochs, dataset_val):
        """
        Train the classifier on the training set `dataset`.

        :param dataset: Training set.
        :param nb_epochs: Number of epochs to train.
        """

        self.model.compile(loss=self.criterion, optimizer=self.optimizer)
        self.model.fit(dataset, epochs=nb_epochs, validation_data=dataset_val)
        # TODO: Vérifier le validation data pour le mettre dans la focntion predict

    def predict(self, dataset):
        """
        Predict the `dataset` with the classifier.

        :param dataset: Testing set.
        """

        # TODO: Implémenter la fonction predict de tensorflowclassifier
        pass
        

    def save(self, filename, path="Networks"):
        """
        Save a model to file in the format of Pytorch framework

        :param filename: Name of the file where is save the model.
        :param path: path where to save the file.
        """

        # TODO: Implémenter la fonction save de tensorflowclassifier
        pass
        

    def load(self, model, path_to_file):
        """
        Load a pytorch saved model (like .pt). You need to define your model in
        the networks_to_load.py file and call it in the function as var 'model'.

        :param model: Architecture of the model to load.
        :param path_to_file: path to the file where is the model to load.
        """

        # TODO: Implémenter la fonction load de tensorflowclassifier
        pass

