# Imports here
import torch
import numpy as np
from torchvision import transforms, datasets, models
from torch import optim
from torch import optim
from collections import OrderedDict
from torch import nn
import time
import matplotlib.pyplot as plt
import json
from PIL import Image
import torch.nn.functional as F
import argparse, sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

test_dir = data_dir + '/test'

data_transforms = {
    'train':transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  
    ]),
    'valid':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(254),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
}

# TODO: Load the datasets with ImageFolder
# use dictionary comprehension
#define directories dictionary
directories = {'train': train_dir, 
               'valid': valid_dir, 
               'test' : test_dir}

image_datasets = {x: datasets.ImageFolder(directories[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=64,shuffle=True)
              for x in  ['train', 'valid', 'test']}