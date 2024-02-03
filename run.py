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

from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss# from torchvision.transforms.v2 import CutMix

# arcface
from evaluation.arcface import ArcFaceModel, ArcFaceSEnet
from evaluation.eval_verification import eval_verification

# new
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
known_images  = torch.stack([transforms(y) for y in known_images])
#Print your shapes here to understand what we have done

# You can use other similarity metrics like Euclidean Distance if you wish
similarity_metric = torch.nn.CosineSimilarity(dim= 1, eps= 1e-6)

# Started:
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

DATA_DIR    = 'content/data/11-785-f23-hw2p2-classification/'
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "dev")
TEST_DIR    = os.path.join(DATA_DIR, "test")

config_file = 'basic_config.yaml'
config = load_config(os.path.join('configs', config_file))

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(224, padding=8, padding_mode='reflect'),
    torchvision.transforms.RandomGrayscale(p=0.12),
    torchvision.transforms.RandAugment(num_ops=2, magnitude=9),
    torchvision.transforms.ToTensor(),
    # normalization
    torchvision.transforms.Normalize(mean=[0.5103,0.4014,0.3508], std=[0.3077,0.2701,0.2591]),
    # Random Erasing (Cutout)
    torchvision.transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)
])


valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5103,0.4014,0.3508], std=[0.3077,0.2701,0.2591])
])

train_dataset   = torchvision.datasets.ImageFolder(TRAIN_DIR, transform= train_transforms)
# Apply cutmix
# train_dataset = CutMix(train_dataset, num_class=7001, beta=1.0, prob=0.5, num_mix=2)    # this is paper's original setting for cifar.

valid_dataset   = torchvision.datasets.ImageFolder(VAL_DIR, transform= valid_transforms)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
                                            dataset     = train_dataset,
                                            batch_size  = config['batch_size'],
                                            shuffle     = True,
                                            num_workers = 6,
                                            pin_memory  = True
                                          )

valid_loader = torch.utils.data.DataLoader(
                                            dataset     = valid_dataset,
                                            batch_size  = config['batch_size'],
                                            shuffle     = False,
                                            num_workers = 5
                                            )

print("Number of classes    : ", len(train_dataset.classes))
print("No. of train images  : ", train_dataset.__len__())
print("Shape of image       : ", train_dataset[0][0].shape)
print("Batch size           : ", config['batch_size'])
print("Train batches        : ", train_loader.__len__())
print("Val batches          : ", valid_loader.__len__())

# load model
model = CustomSEResNet(dropout=True, dropout_prob=0.3).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
# Apply Cutmix loss
# criterion = CutMixCrossEntropyLoss(True)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0)
optimizer = torch.optim.RAdam(model.parameters(), lr=0.0001, weight_decay= 0)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0,
                                                          last_epoch=-1, verbose=False)

# Clean CUDA OOM error
gc.collect()
torch.cuda.empty_cache()

# Load pretrained model
checkpoint = torch.load(os.path.join('custom_se_resNet_dr.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
# eval_verification(unknown_dev_images, known_images, known_paths, model, similarity_metric, config['batch_size'], mode='val')

# change feature map
# arc_model = ArcFaceModel(model, margin=0.49, scaler=49,
#                      embedding_size=1024, num_classes=7001).to(DEVICE)

# model = ArcFaceSEnet(model, num_classes=7001).to(DEVICE)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
for g in optimizer.param_groups:
    prior_lr = g['lr']
    prior_weight_decay = g['weight_decay']
    print(f'Best prior lr: {prior_lr}, and weight_decay: {prior_weight_decay}')
    g['lr'] = 0.005
    # 0.00001 -> 0.00005 -> 0.0001 -> 0.0003 (98% training) -> 0.0006
    g['weight_decay'] = 0

# optimizer = torch.optim.RAdam(model.parameters(), lr=0.005, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0,
                                                          last_epoch=-1, verbose=False)

epoch = checkpoint['epoch']
best_val_acc = checkpoint['val_acc']
train_acc = checkpoint['train_acc']
print(f'Best validation accuracy is: {best_val_acc} at {epoch} epoch with training acc: {train_acc}')

# Create your wandb run
wandb.login(key="c06ce00d5f99f931dfcbf6b470908fe8de32451c")
run = wandb.init(
                name = "CustomSEResNet",
                reinit = True,
                id = 'zvrx1e9d',
                resume = "must",
                project = "hw2p2-ablations",
                config = config
            )

# best_val_acc = None
# if best_val_acc is None:
#     best_val_acc = 0.8

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# checkpoint = torch.load(os.path.join('arcface_embedding.pth'))
# model.load_state_dict(checkpoint['model_state_dict'])

for epoch in range(50):

    curr_lr = float(optimizer.param_groups[0]['lr'])
    print(curr_lr)

    train_acc, train_loss = train(model, train_loader, config=config, optimizer=optimizer,
                                  criterion=criterion, lr_scheduler=None,
                                  device=DEVICE)

    print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
                epoch + 1,
                config['epochs'],
                train_acc,
                train_loss,
                curr_lr))

    # model = ArcFaceSEnet(model, num_classes=7001).to(DEVICE)
    val_acc, val_loss = validate(model, valid_loader, config=config, criterion=criterion, device=DEVICE)
    # pred_id_strings, eva_accuracy = eval_verification(unknown_dev_images, known_images, known_paths, model,
    #                                                   similarity_metric, config['batch_size'], mode='val')

    #
    lr_scheduler.step()
    #
    print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

    wandb.log({"train_loss": train_loss, 'train_Acc': train_acc, 'validation_Acc': val_acc,
               'validation_loss': val_loss, "learning_Rate": curr_lr})

    # wandb.log({"train_loss":train_loss, "learning_Rate": curr_lr})
    # if arc_loss >= train_loss:
    #     arc_loss = train_loss
    #     print("Saving model")
    #     torch.save({'model_state_dict': arc_model.state_dict()}, './arcface_embedding.pth')

    # Save model
    if val_acc >= best_val_acc:
        #path = os.path.join(root, model_directory, 'checkpoint' + '.pth')
        print("Saving model")
        torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'scheduler_state_dict':lr_scheduler.state_dict(),
                  'val_acc': val_acc,
                  'train_acc': train_acc,
                  'epoch': epoch}, './custom_se_resNet_dr_cons.pth')
        best_val_acc = val_acc
        # wandb.save('hw2_classification_2_more.pth')

    if curr_lr < 0.000001:
        optimizer.param_groups[0]['lr'] = 0.000066
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0,
                                                                  last_epoch=-1, verbose=False)

print(f'now the best val: {best_val_acc}')
run.finish()

