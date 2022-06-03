###############################################################################
# Packages                                                                    #
###############################################################################
import torchvision
import torch
import torchvision.transforms as transforms


batch_size = 64
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((160, 160)),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
     )

trainset = torchvision.datasets.ImageFolder(
        "imagenette2-160/train", transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(
    "imagenette2-160/val", transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)
