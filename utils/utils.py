###############################################################################
# Packages                                                                    #
###############################################################################

import argparse
import yaml
from yaml.loader import SafeLoader
import torch

###############################################################################
# Functions                                                                   #
###############################################################################

def config_args():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('config_file', type=str,
                     help='Path to the config file')
    return parser.parse_args() 


def get_config(config_file):
    with open(config_file) as f:
        Config = yaml.load(f, Loader=SafeLoader)
    return Config

def load_model(model, path, weights_loaded=True):
    """
    Load the model architectures and weights
    """
    # You can customize this function to load your own model based on model name.
    model_ori = eval(model)()
    model_ori.eval()
    print(model_ori)

    if not weights_loaded:
        return model_ori

    if path is not None:
        sd = torch.load(path, map_location=torch.device('cpu'))
        if 'state_dict' in sd:
            sd = sd['state_dict']
        if isinstance(sd, list):
            sd = sd[0]
        if not isinstance(sd, dict):
            raise NotImplementedError("Unknown model format, please modify model loader yourself.")
        model_ori.load_state_dict(sd)
    else:
        print("Warning: pretrained model path is not given!")

    return model_ori

# def config_args():
#     parser = argparse.ArgumentParser(description='Optional app description')
#     # Required positional argument
#     parser.add_argument('pos_arg', type=int,
#                     help='A required integer positional argument')

#     # Optional positional argument
#     parser.add_argument('opt_pos_arg', type=int, nargs='?',
#                     help='An optional integer positional argument')

#     # Optional argument
#     parser.add_argument('--root_train', type=int, help='An optional root to train dataset')
#     parser.add_argument('--root_test', type=int, help='An optional root to test dataset')

#     # Switch
#     parser.add_argument('--switch', action='store_true',
#                     help='A boolean switch')

#     return parser.parse_args()     

# args = config_args()           

# print("Argument values:")
# print(args.pos_arg)
# print(args.opt_pos_arg)
# print(args.opt_arg)
# print(args.switch)