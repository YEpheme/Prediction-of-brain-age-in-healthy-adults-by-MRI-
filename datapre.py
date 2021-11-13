!pip install pydicom
import os
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot

base_dir = '/home/tione/notebook/taop-2021/100002'
df = pd.read_csv(os.path.join(base_dir,'train2_data_info.csv'),encoding='utf-8')
df.head(10)

load_paths, std_paths, cur_dcm_paths = [], [], []
for i in range(0,df.shape[0],5):
    if df.loc[i, 'id_patient'] == 'CNBD_02316':
        continue
    load_path = '/home/tione/notebook/Data/T1_Raw/'+df.loc[i, 'id_patient']
    std_path = '/home/tione/notebook/Data/T1_Standard/'+df.loc[i, 'id_patient']
    cur_dcm_path = os.path.join(base_dir, df.loc[i+1,'file_path'][1:])
    load_paths.append(load_path)
    std_paths.append(std_path)
    cur_dcm_paths.append(cur_dcm_path)

img_means = np.zeros(const_pixel_dims)
img_var = np.zeros(const_pixel_dims)
for index in range(len(cur_dcm_paths)):
    l_dcm_paths = []
    for root, subdirs, files in os.walk(cur_dcm_paths[index]):
        for fname in files:
            if ".dcm" in fname.lower():
                l_dcm_paths.append(os.path.join(root, fname))
    arr_dcms = np.zeros(const_pixel_dims, dtype=ref_dcm.pixel_array.dtype)
    for idx, dcm_path in enumerate(l_dcm_paths):
        ds = pydicom.read_file(dcm_path)
        arr_dcms[:, :, idx] = ds.pixel_array
    np.save(load_paths[index],arr_dcms)
    img_means += arr_dcms
img_means = img_means/len(cur_dcm_paths)
for index in range(len(cur_dcm_paths)):
    img = np.load(load_paths[index]+'.npy')
    img_var += (img - img_means)**2
img_var = img_var/len(cur_dcm_paths)
img_var = np.sqrt(img_var+0.0001)
for index in range(len(cur_dcm_paths)):
    img = np.load(load_paths[index]+'.npy')
    img_std = (img-img_means)/img_var
    np.save(std_paths[index],img_std)
    

df = pd.read_csv(os.path.join(base_dir,'test2_data_info.csv'),encoding='cp936')
load_paths, std_paths, cur_dcm_paths = [], [], []
for i in range(0,df.shape[0],5):
    if df.loc[i, 'id_patient'] == 'CNBD_02316':
        continue
    load_path = '/home/tione/notebook/Data/T1_Raw/'+df.loc[i, 'id_patient']
    std_path = '/home/tione/notebook/Data/T1_Standard/'+df.loc[i, 'id_patient']
    cur_dcm_path = os.path.join(base_dir, df.loc[i+1,'file_path'][1:])
    load_paths.append(load_path)
    std_paths.append(std_path)
    cur_dcm_paths.append(cur_dcm_path)
for index in range(len(cur_dcm_paths)):
    l_dcm_paths = []
    for root, subdirs, files in os.walk(cur_dcm_paths[index]):
        for fname in files:
            if ".dcm" in fname.lower():
                l_dcm_paths.append(os.path.join(root, fname))
    arr_dcms = np.zeros(const_pixel_dims, dtype=ref_dcm.pixel_array.dtype)
    for idx, dcm_path in enumerate(l_dcm_paths):
        ds = pydicom.read_file(dcm_path)
        arr_dcms[:, :, idx] = ds.pixel_array
    np.save(load_paths[index],arr_dcms)
for index in range(len(cur_dcm_paths)):
    img = np.load(load_paths[index]+'.npy')
    img_std = (img-img_means)/img_var
    np.save(std_paths[index],img_std)