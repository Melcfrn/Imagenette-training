###############################################################################
# Packages                                                                    #
###############################################################################
from pickletools import optimize
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from Loaders.pytorch_loader import PytorchLoader
from DefModels.model import Model, ModelTF
from Classifiers.pytorch_classifier import PytorchClassifier
from DefModels.resnet_model import ResNet5, ResNet9, BasicBlock
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input

from Loaders.tensorflow_loader import TensorflowLoader
from Classifiers.tensorflow_classifier import TensorflowClassifier

###############################################################################
# Dataset + parameters                                                        #
###############################################################################

batch_size = 32
nb_epochs = 20

size = 160

root_to_train = f"imagenette2-{size}/train"
root_to_test = f"imagenette2-{size}/val"

########################
# PYTORCH              #
########################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((size, size)),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
     )

trainloader = PytorchLoader(location=root_to_train, transform=transform,
                     batch_size=batch_size, shuffle=True).get_dataset()
testloader = PytorchLoader(location=root_to_test, transform=transform,
                    batch_size=batch_size, shuffle=False).get_dataset()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# def resnet2b():
#     return ResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

# def resnet4b():
#     return ResNet9(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")
net = Model(3, [32, 64, 128], [512], 10, activation='relu')
# net = ResNet9(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
print(net)

########################
# TENSORFLOW           #
########################

# data_augmentation = keras.Sequential(
#     [
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#     ]
# )

# trainloader = TensorflowLoader(location=root_to_train, size=size)

# # device ????????
# # TODO: Ajouter l'utilisation cpu/gpu pour tensorflow

# net = ModelTF(image_size=size)

# criterion=keras.losses.categorical_crossentropy,
# optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

# net.summary()

###############################################################################
# Training                                                                    #
###############################################################################

########################
# PYTORCH              #
########################

classifier = PytorchClassifier(model=net,
                               criterion=criterion,
                               optimizer=optimizer,
                               batch_size=batch_size,
                               device=device
                               )

classifier.fit(dataset=trainloader, nb_epochs=nb_epochs)
# print("Accuracy on trainset to watch overfitting : \n")
# classifier.predict(dataset=trainloader)
# print("Accuracy on testset : \n")
classifier.predict(dataset=testloader)
classifier.save(filename=f"first_model_{size}.pt")

########################
# TENSORFLOW           #
########################

# classifier = TensorflowClassifier(model=net,
#                                   criterion=criterion,
#                                   optimizer=optimizer,
#                                   batch_size=batch_size
#                                   )

# classifier.fit(dataset=trainloader, nb_epochs=nb_epochs)

# # TODO: Ajouter predict et save