###############################################################################
# Packages                                                                    #
###############################################################################

import torch
import os
from time import sleep
from tqdm import tqdm
from networks_to_load import cifar_conv

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
        batch_size=64,
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
            loop = tqdm(dataset, ncols=100, initial=1)
            # running_loss = 0.0
            for i, data in enumerate(loop, 0):
                inputs, labels = data[0].to(
                    self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                loop.set_description(f"Epoch [{epoch}/{nb_epochs}]")
                loop.set_postfix(loss=loss.item(), acc=(predicted == labels).sum().item()/total)
                # if i % 100 == 99:
                #     print('[{},{:5d}] loss: {:.3f}'.format(
                #         epoch + 1, i + 1, running_loss / 100))
                #     running_loss = 0.0
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

        :param filename: Name of the file where is save the model.
        :param path: path where to save the file.
        """

        self.model.eval()
        if path is None:
            complete_path = filename
        else:
            complete_path = os.path.join(path, filename)

        torch.save(self.model.state_dict(), complete_path)
        print('Model saved to: {}'.format(complete_path))

    def load(self, model, path_to_file):
        """
        Load a pytorch saved model (like .pt). You need to define your model in
        the networks_to_load.py file and call it in the function as var 'model'.

        :param model: Architecture of the model to load.
        :param path_to_file: path to the file where is the model to load.
        """
        net = model
        net.load_state_dict(torch.load(path_to_file))
        net.eval()
        return net

#     def printer(self, running_loss, epoch, nb_epochs):

# for i in tqdm(range(10)):
#     a = 2
#         if i % 100 == 99:
#             print('[{},{:5d}] loss: {:.3f}'.format(
#                 epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0