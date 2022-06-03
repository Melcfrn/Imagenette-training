###############################################################################
# Packages                                                                    #
###############################################################################

import torch
import os

###############################################################################
# Pytorch Classifier                                                          #
###############################################################################


class PytorchClassifier():
    """
    This class implements a neural network classifier
    with the PyTorch framework
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        batch_size,
        device="cpu"
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.model.to(self.device)

    def fit(self, dataset, nb_epochs):
        """
        Train the classifier on the training set `dataset`.

        :param dataset: Training set.
        :param nb_epochs: Number of epochs to train.
        """

        self.model.train()
        for epoch in range(nb_epochs):
            running_loss = 0.0
            for i, data in enumerate(dataset, 0):
                inputs, labels = data[0].to(
                    self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % self.batch_size == self.batch_size - 1:
                    print('[{},{:5d}] loss: {:.3f}'.format(
                        epoch + 1, i + 1, running_loss / self.batch_size))
                    running_loss = 0.0
        print('Finished Training')

    def predict(self, dataset):
        """
        Predict the `dataset` with the classifier.

        :param dataset: Testing set.
        """

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataset:
                images, labels = data[0].to(
                    self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the {} test images: {} %'.format(
            total, 100 * correct // total))

    def save(self, filename, path=None):
        """
        Save a model to file in the format of Pytorch framework
        """

        self.model.eval()
        if path is None:
            complete_path = filename
        else:
            complete_path = os.path.join(path, filename)

        torch.save(self.model.state_dict(), complete_path)
        print('Model saved to: {}'.format(complete_path))
