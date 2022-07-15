###############################################################################
# Packages                                                                    #
###############################################################################

import torch
import torch.nn as nn

###############################################################################
# Some functions                                                              #
###############################################################################

def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['sigmoid', nn.Sigmoid()]
    ])
    
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        activations[activation],
        nn.Conv2d(out_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations[activation],
        nn.MaxPool2d(2, 2),
        # PrintSize()
    )

def dec_block(in_f, out_f, activation='relu'):
    activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['relu', nn.ReLU()],
            ['sigmoid', nn.Sigmoid()]
    ])

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_f, out_f),
        activations[activation]
    )

###############################################################################
# Some classes                                                                #
###############################################################################

class MyEncoder(nn.Module):
    def __init__(self, enc_sizes, *args, **kwargs):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f, kernel_size=3, padding=1, stride=1, *args, **kwargs) 
                       for in_f, out_f in zip(enc_sizes, enc_sizes[1:])])
        
    def forward(self, x):
        return self.conv_blocks(x)
        
class MyDecoder(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f, 'relu') 
                       for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes)

    def forward(self, x):
        return self.dec_blocks(x)

###############################################################################
# Model                                                                       #
###############################################################################

class Model(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes,  n_classes, activation='relu'):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [self.enc_sizes[-1] * 20 ** 2, *dec_sizes]

        self.encoder = MyEncoder(self.enc_sizes, activation=activation)

        self.decoder = MyDecoder(self.dec_sizes, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        
        # print(x.shape)
        
        x = self.decoder(x)
        
        return x


# class Model(nn.Module):
#     """
#     This class implements a neural network from PyTorch framework
#     """

#     def __init__(self):
#         super().__init__()
#         self.kernel_size = 5
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=self.kernel_size, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(32),
#             nn.Dropout(0.3),

#             nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(64),
#             nn.Dropout(0.3),

#             nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(128),
#             nn.Dropout(0.3),

#             nn.Flatten(),
#             nn.Linear(128 * 38 * 38, 100),
#             nn.ReLU(),
#             nn.BatchNorm1d(100),
#             nn.Dropout(0.3),
#             nn.Linear(100, 10)
#         )

#     def forward(self, x):
#         return self.network(x)
