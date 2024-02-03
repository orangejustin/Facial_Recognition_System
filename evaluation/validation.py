from eval_verification import eval_verification
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm.auto import tqdm
import torchvision #Augmentations
from PIL import Image
import torch.nn as nn
import os
import sys
sys.path.append('/home/zechengli/HW2P2')
from models.cnn_residual import CustomResNet
import gc
from torchvision.ops import sigmoid_focal_loss

# New
import torch
from torchsummary import summary
import torchvision #Augmentations
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import wandb
import matplotlib.pyplot as plt
import timm

from utils.util import load_config
from models import CustomResNet, CustomResNet_dr, ResNetMHSA, CustomSEResNet
from train import *

# TODO: simplified
from PIL import ImageEnhance
from torchvision.transforms import functional as F


DEVICE = 'cuda'
batch_size = 128

# This obtains the list of known identities from the known folder
known_regex = "./content/data/11-785-f23-hw2p2-verification/known/*/*"
known_paths = [i.split('/')[-2] for i in sorted(glob.glob(known_regex))]

# Obtain a list of images from unknown folders
unknown_dev_regex = "./content/data/11-785-f23-hw2p2-verification/unknown_dev/*"
unknown_test_regex = "./content/data/11-785-f23-hw2p2-verification/unknown_test/*"

# We load the images from known and unknown folders
unknown_dev_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_dev_regex)))]
unknown_test_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_test_regex)))]
known_images = [Image.open(p) for p in tqdm(sorted(glob.glob(known_regex)))]

# Why do you need only ToTensor() here?
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

unknown_dev_images = torch.stack([transforms(x) for x in unknown_dev_images])
unknown_test_images = torch.stack([transforms(x) for x in unknown_test_images])
known_images  = torch.stack([transforms(y) for y in known_images ])
#Print your shapes here to understand what we have done

# You can use other similarity metrics like Euclidean Distance if you wish
similarity_metric = torch.nn.CosineSimilarity(dim= 1, eps= 1e-6)
num_classes = len(known_paths)
# print(num_classes) # not the # of data point 7001

eval_verification(unknown_dev_images, known_images, known_paths, model,
                                                      similarity_metric, 128, mode='val')