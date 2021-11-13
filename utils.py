import os
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
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

base_dir = '/home/tione/notebook/taop-2021/100002'

def getData(csv_path):
    img_path, img_label = [], []
    df = pd.read_csv(csv_path, encoding='utf-8')
    for i in range(0,df.shape[0],5):
        if df.loc[i, 'id_patient'] == 'CNBD_02316':
            continue
        if df.loc[i, 'id_patient'] == 'CNBD_02136':
            img_label.append(57.0)
        else:
            img_label.append(df.loc[i, 'id_age'])
        path = '/home/tione/notebook/Data/T1_Standard/'+df.loc[i, 'id_patient']+'.npy'
        img_path.append(path)
    return img_path, img_label

class MyDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
            
    def __getitem__(self, index):
        img = np.load(self.img_path[index])
        img = img.reshape((1,)+img.shape)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.img_label[index]
    
    def __len__(self):
        return len(self.img_path)
    
class TestDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
            
    def __getitem__(self, index):
        img = np.load(self.img_path[index])
        img = img.reshape((1,)+img.shape)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_path)

def train(train_loader, model, criterion, optimizer):
    train_loss = []
    model.train()
    for x, y in train_loader:
        batch_x  = Variable(x.cuda())
        batch_y = Variable(y.cuda())
        output = model(batch_x.float())
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.cpu().item())
    return np.mean(train_loss)

def validate(val_loader, model, criterion):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for x,y in val_loader:
            batch_x = Variable(x.cuda())
            batch_y = Variable(y.cuda())
            output = model(batch_x.float())
            loss=criterion(output, batch_y)
            val_loss.append(loss.cpu().item())
    return np.mean(val_loss)

