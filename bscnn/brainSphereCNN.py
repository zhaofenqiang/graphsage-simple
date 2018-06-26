#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:18:30 2018

@author: zfq
"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init


class BrainSphereCNN(nn.Module):

    def __init__(self, num_classes, neigh_orders):
        super(BrainSphereCNN, self).__init__()

        self.feature_dims = [3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, num_classes]
        
        sequence = []
        for l in range(len(self.feature_dims) - 2):
            nfeature_in = self.feature_dims[l]
            nfeature_out = self.feature_dims[l + 1]      
            sequence.append(BSLayer(nfeature_in, nfeature_out, neigh_orders))
            sequence.append(nn.BatchNorm1d(nfeature_out))
            sequence.append(nn.ReLU())
            
        sequence.append(BSLayer(self.feature_dims[-2], self.feature_dims[-1], neigh_orders))
        self.sequential = nn.Sequential(*sequence)
        
    def forward(self, x):
        x = self.sequential(x)
        return x
    
class BSLayer(nn.Module):

    def __init__(self, in_feats, out_feats, neigh_orders):
        super(BSLayer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders
        
        self.weight = nn.Linear(7 * in_feats, out_feats)
        
    def forward(self, x):
       
        mat = x[self.neigh_orders].view(len(x), 7*self.in_feats)
                
        out_features = self.weight(mat)
        return out_features
