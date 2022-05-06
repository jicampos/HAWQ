"""
    Fixing issue with loading checkpoints. 
"""

import os, sys
sys.path.insert(1, os.path.abspath('.'))

import torch
from utils import q_jettagger_model
from quant_train import save_checkpoint

def create_checkpoint(model):
    print('Model')
    print('-----------------------------------------------------------------------------')
    print(model)
    print('-----------------------------------------------------------------------------')

    save_checkpoint({
                'epoch': 0,
                'arch': 'Q_JetTagger',
                'state_dict': model.state_dict(),
                'best_acc1': '0.0',
                'optimizer': {},
            }, False, './')

def print_checkpoint(filename):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))

    for name, val in checkpoint.items():
        print(name, type(val))

def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))['state_dict']
    model_state_dict = model.state_dict()
    
    modified_dict = {}
    
    for key, value in checkpoint.items():
        if model_state_dict[key].shape != value.shape:
            value = torch.tensor([value])
        modified_dict[key] = value
    
    model.load_state_dict(modified_dict, strict=False)


if __name__ == '__main__':

    filename = 'checkpoint.pth.tar'

    print_checkpoint(filename)

    # model = q_jettagger_model()
    # load_checkpoint(model, filename)

    bp = 0 

