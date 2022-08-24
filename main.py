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
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input

from Loaders.tensorflow_loader import TensorflowLoader
from Classifiers.tensorflow_classifier import TensorflowClassifier
from utils.utils import get_config, config_args, load_model

###############################################################################
# Dataset + parameters                                                        #
###############################################################################
args = config_args() 

Config = get_config(args.config_file)

# batch_size = 32
# nb_epochs = 20

# size = 160

# root_to_train = f"imagenette2-{size}/train"
# root_to_test = f"imagenette2-{size}/val"

########################
# PYTORCH              #
########################

if Config["model"]["framework"] == "Pytorch":

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize(Config["transformations"]["resize"]),
    transforms.Normalize(Config["transformations"]["means"], Config["transformations"]["std"])]
    )

    if Config["parameters"]["device"] == 'cpu':
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # TODO: Mieux ordonner les boucles suivantes
    if Config["aim"]["train"]:
        trainloader = PytorchLoader(location=Config["data"]["root_train"], transform=transform,
                        batch_size=Config["parameters"]["batch_size"], shuffle=Config["parameters"]["shuffle"]).get_dataset()

        # def resnet2b():
        #     return ResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

        # def resnet4b():
        #     return ResNet9(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")

        # net = ResNet9(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")
        net = load_model(Config["model"]["type"], Config["model"]["path"])
        # net = Model(3, [32, 64, 128], [512], 10, activation='relu') 
        if Config["parameters"]["criterion"] == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Unexpected loss {Config["parameters"]["criterion"]}. Only CrossEntropyLoss are available.')
        
        if Config["parameters"]["optimizer"]["type"] == "Adam":
            optimizer = optim.Adam(net.parameters(), lr=Config["parameters"]["optimizer"]["lr"])
        else:
            raise ValueError(f'Unexpected optimizer {Config["parameters"]["optimizer"]["type"]}. Only Adam are available.')

        print(net)
        
        classifier = PytorchClassifier(model=net,
                               criterion=criterion,
                               optimizer=optimizer,
                               batch_size=Config["parameters"]["batch_size"],
                               device=device
                               )

        classifier.fit(dataset=trainloader, nb_epochs=Config["parameters"]["nb_epochs"])

    if Config["aim"]["test"] and Config["aim"]["train"]:
        testloader = PytorchLoader(location=Config["data"]["root_test"], transform=transform,
                        batch_size=Config["parameters"]["batch_size"], shuffle=False).get_dataset()
        
        classifier.predict(dataset=testloader)

    if Config["aim"]["test"] and not Config["aim"]["train"]:
        raise NotImplementedError(f'Unexpected road. Testing without training before is not yet available.')
        # TODO: Implémenter le cas où on veut juste tester

    if Config["aim"]["inference"]:
        raise NotImplementedError(f'Unexpected aim {Config["aim"]["inference"]}. Inference is not yet available.')
        # TODO: Implémenter l'inférence
    
    if Config["aim"]["transfer_learning"]:
        raise NotImplementedError(f'Unexpected aim {Config["aim"]["transfer_learning"]}. Transfer learning is not yet available.')
        # TODO: Implémenter le transfer learning

    if Config["aim"]["save"]:
        classifier.save(filename=Config["model"]["save_name"])

########################
# TENSORFLOW           #
########################
elif Config.model.framework == "Tensorflow":
    raise NotImplementedError(f'Unexpected framework {Config["model"]["framework"]}. Tensorflow is not yet available.')
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

# classifier = PytorchClassifier(model=net,
#                                criterion=criterion,
#                                optimizer=optimizer,
#                                batch_size=batch_size,
#                                device=device
#                                )

# classifier.fit(dataset=trainloader, nb_epochs=nb_epochs)
# # print("Accuracy on trainset to watch overfitting : \n")
# # classifier.predict(dataset=trainloader)
# # print("Accuracy on testset : \n")
# classifier.predict(dataset=testloader)
# classifier.save(filename=f"first_model_{size}.pt")

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

else:
    raise ValueError(f'Unexpected framework {Config["model"]["framework"]}. Only Pytorch and Tensorflow are available.')