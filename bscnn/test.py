#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:48:12 2018

@author: zfq
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import torchvision
import scipy.io as sio 
import numpy as np
import glob
import os

from sklearn.metrics import f1_score
from bscnn.brainSphereCNN import BrainSphereCNN

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(self.root, '*.mat')))    
        self.transform = transform
        
    def __getitem__(self, index):
        file = self.files[index]
        feats = sio.loadmat(file)
        feats = feats['data']
        feats = (feats - np.tile(np.min(feats,0), (len(feats),1)))/(np.tile(np.max(feats,0), (len(feats),1)) - np.tile(np.min(feats,0), (len(feats),1)))
        feats = feats - np.tile(np.mean(feats, 0), (len(feats), 1))
        label = sio.loadmat(file[:-4] + '.label')
        label = label['label']    
        label = np.squeeze(label)
        label = label - 1
        return feats.astype(np.float32), label.astype(np.long)

    def __len__(self):
        return len(self.files)


learning_rate = 0.1
momentum = 0.9
wd = 0.0001
batch_size = 1
test_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/test'
test_dataset = BrainSphere(test_dataset_path)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

model = BrainSphereCNN(36)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda()
model.load_state_dict(torch.load('/home/zfq/graphsage-simple/state.pkl'))
model.eval()
test_total_correct = 0
for batch_idx, (data, target) in enumerate(test_dataloader):
    data = data.squeeze()
    target = target.squeeze()
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        prediction = model(data)
    test_total_correct += prediction.max(1)[1].eq(target).long().cpu().sum() 
test_acc = test_total_correct.item() / (len(test_dataloader) * 10242)    
print("Test ACC = ", test_acc)
