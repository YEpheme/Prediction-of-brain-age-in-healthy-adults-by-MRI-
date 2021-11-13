import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from utils import getData, MyDataset, train, validate
import ConvMixer

train_path = '/home/tione/notebook/Code/Train/kfold/Label-train-fold1.csv'
val_path = '/home/tione/notebook/Code/Train/kfold/Label-eval-fold1.csv'

train_paths, train_label = getData(train_path)
val_paths, val_label = getData(val_path)

train_loader = DataLoader(MyDataset(train_paths, train_label), batch_size = 24, shuffle = False)
val_loader = DataLoader(MyDataset(val_paths, val_label), batch_size = 24, shuffle = False)

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

model = ConvMixer.ConvMixer(256, 4, kernel_size=9, patch_size=32, n_classes=1)
print(model)

criterion = nn.L1Loss(reduction = 'mean')
criterion = criterion.cuda()
number_epoch = 10

model = model.cuda()
    
optimizer = torch.optim.Adam(model.parameters(), 0.005)
torch.backends.cudnn.enabled = False
train_mae_history, val_mae_history = [], []

for epoch in range(number_epoch):
    train_loss = train(train_loader, model, criterion, optimizer)
    val_loss = validate(val_loader, model, criterion)
    train_mae_history.append(train_loss)
    val_mae_history.append(val_loss)
    print('Epoch:',epoch,'Train Loss:',train_loss,'Eval Loss:',val_loss)
    torch.save(model, r'/home/tione/notebook/Model/ConvMixer_'+str(epoch)+'.pth')


 