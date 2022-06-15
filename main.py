###############################################################################
# Packages                                                                    #
###############################################################################
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from loader import Loader
from model import Model
from pytorch_classifier import PytorchClassifier

###############################################################################
# Dataset + parameters                                                        #
###############################################################################

batch_size = 64
nb_epochs = 20
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((320, 320)),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
     )
root_to_train = "imagenette2-320/train"
root_to_test = "imagenette2-320/val"

trainloader = Loader(location=root_to_train, transform=transform,
                     batch_size=batch_size, shuffle=True).get_dataset()
testloader = Loader(location=root_to_test, transform=transform,
                    batch_size=batch_size, shuffle=False).get_dataset()

net = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

###############################################################################
# Training                                                                    #
###############################################################################

classifier = PytorchClassifier(model=net,
                               criterion=criterion,
                               optimizer=optimizer,
                               batch_size=batch_size
                               )

classifier.fit(dataset=trainloader, nb_epochs=nb_epochs)
classifier.predict(dataset=testloader)
classifier.save(filename="first_model_320.pt")
