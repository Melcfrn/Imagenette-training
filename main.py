###############################################################################
# Packages                                                                    #
###############################################################################
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from loader import Loader
from model import Model
from pytorch_classifier import PytorchClassifier
from resnet_model import ResNet5, ResNet9, BasicBlock

###############################################################################
# Dataset + parameters                                                        #
###############################################################################

batch_size = 64
nb_epochs = 30

size = 160
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((size, size)),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
     )
root_to_train = f"imagenette2-{size}/train"
root_to_test = f"imagenette2-{size}/val"

trainloader = Loader(location=root_to_train, transform=transform,
                     batch_size=batch_size, shuffle=True).get_dataset()
testloader = Loader(location=root_to_test, transform=transform,
                    batch_size=batch_size, shuffle=False).get_dataset()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# def resnet2b():
#     return ResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

# def resnet4b():
#     return ResNet9(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")
net = ResNet9(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

###############################################################################
# Training                                                                    #
###############################################################################

classifier = PytorchClassifier(model=net,
                               criterion=criterion,
                               optimizer=optimizer,
                               batch_size=batch_size,
                               device=device
                               )

classifier.fit(dataset=trainloader, nb_epochs=nb_epochs)
print("Accuracy on trainset to watch overfitting : \n")
classifier.predict(dataset=trainloader)
print("Accuracy on testset : \n")
classifier.predict(dataset=testloader)
classifier.save(filename=f"first_model_{size}.pt")
