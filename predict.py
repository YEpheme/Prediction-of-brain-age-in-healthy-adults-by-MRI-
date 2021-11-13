import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from utils import getData, MyDataset, TestDataset

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

model = torch.load(r'/home/tione/notebook/Model/ConvMixer_10.pth')

base_dir = '/home/tione/notebook/taop-2021/100002'
test_csv_path = '/home/tione/notebook/taop-2021/100002/test2_data_info.csv'
train_csv_path = '/home/tione/notebook/taop-2021/100002/train2_data_info.csv'
df = pd.read_csv(test_csv_path, encoding='cp936')
df.head(10)

T1_paths = []
id_type, id_project, id_patient, id_exam, id_series, id_image, id_doctor, id_age =[], [], [], [], [], [], [], []
for i in range(0,df.shape[0],5):
        id_type.append(df.loc[i+3, 'id_type'])
        id_project.append(df.loc[i+3, 'id_project'])
        id_exam.append(df.loc[i+2, 'id_exam'])
        id_patient.append(df.loc[i+3, 'id_patient'])
        id_series.append(df.loc[i+2, 'id_series'])
        id_image.append(df.loc[i+3, 'id_image'])
        id_doctor.append(df.loc[i+3, 'id_doctor'])
        T1_path = '/home/tione/notebook/Data/T1_Standard/'+df.loc[i, 'id_patient']+'.npy'
        T1_paths.append(T1_path)
test_loader = DataLoader(TestDataset(T1_paths), batch_size = 1, shuffle = False)

model.eval()
with torch.no_grad():
    for x in test_loader:
        batch_x  = Variable(x.cuda())
        output = model(batch_x.float())
        id_age.append(output.item())

data = {'id_type':id_type, 'id_project':id_project,'id_patient':id_patient, 'id_exam':id_exam, 'id_series':id_series, 'id_image':id_image, 'id_doctor':id_doctor, 'id_age':id_age }
frame = pd.DataFrame(data)

frame.to_csv('02_results.csv', encoding='cp936', index=False)
import time
import datetime
t = time.time()
save_dir = '/home/tione/notebook/taop-2021-result/02_results/'+str(int(t))
os.makedirs(save_dir)