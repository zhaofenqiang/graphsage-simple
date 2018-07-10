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
from tensorboardX import SummaryWriter


writer = SummaryWriter('log/adam')

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
momentum = 0.99
wd = 0.0001
batch_size = 1
train_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/train'
val_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/val'
train_dataset = BrainSphere(train_dataset_path)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(val_dataset_path)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model = BrainSphereCNN(36)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=momentum, weight_decay=wd)
criterion = nn.CrossEntropyLoss()

def train_step(data, target):
    model.train()
    data, target = data.cuda(), target.cuda()

    prediction = model(data)
    
    loss = criterion(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    correct = prediction.max(1)[1].eq(target).long().cpu().sum()

    return loss.item(), correct.item()

def get_learning_rate(epoch):
    limits = [3, 5, 10, 15, 20]
    lrs = [1, 0.8, 0.5, 0.2, 0.1, 0.01]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate

def val_during_training():
    model.eval()

    val_total_correct = 0
    for batch_idx, (data, target) in enumerate(val_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            prediction = model(data)
        val_total_correct += prediction.max(1)[1].eq(target).long().cpu().sum() 
    val_acc = val_total_correct.item() / (len(val_dataloader) * 10242)    
    
    
    train_total_correct = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            prediction = model(data)
        train_total_correct += prediction.max(1)[1].eq(target).long().cpu().sum() 
    train_acc = train_total_correct.item() / (len(train_dataloader) * 10242)    
    
    return val_acc, train_acc

for epoch in range(300):

    val_acc, train_acc = val_during_training()
    print("Val ACC = ", val_acc, "Train ACC = ", train_acc)
    writer.add_scalars('data/Acc', {'train': train_acc, 'val': val_acc}, epoch)

    lr = get_learning_rate(epoch)
    print("learning rate = {} and batch size = {}".format(lr, train_dataloader.batch_size))
    for p in optimizer.param_groups:
        #print(p)
        p['lr'] = lr

    total_loss = 0
    total_correct = 0
    
#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        loss, correct = train_step(data, target)

        print("[{}:{}/{}]  LOSS={:.4}  ACC={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss, correct/target.size()[0]))
        
        writer.add_scalar('Train/Loss', loss, epoch*70 + batch_idx)
        
    if epoch % 5 == 0 :
        torch.save(model.state_dict(), os.path.join("state.pkl"))
