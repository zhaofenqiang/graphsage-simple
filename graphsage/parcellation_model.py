import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import glob
import math
import os
from sklearn.metrics import f1_score
from collections import defaultdict
import scipy.io as sio 

from graphsage.parcellation_encoders import Encoder
from graphsage.parcellation_aggregators import MeanAggregator
from tensorboardX import SummaryWriter

writer = SummaryWriter('log/first')

learning_rate = 0.1
wd = 0.0001
batch_size = 4
train_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/train'
val_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/val'
files = sorted(glob.glob(os.path.join(train_dataset_path, '*.mat'))) 
val_files = sorted(glob.glob(os.path.join(val_dataset_path, '*.mat'))) 

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes):
        super(SupervisedGraphSage, self).__init__()

        adj_mat = sio.loadmat('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat.mat')
        adj_mat = adj_mat['adj_mat']
        non_zero_indices = np.nonzero(adj_mat)
        adj_lists = defaultdict(set)
        for i in range(len(non_zero_indices[0])):
            adj_lists[non_zero_indices[0][i]].add(non_zero_indices[1][i])

        self.agg1 = MeanAggregator(features=None)
        #features, feature_dim, embed_dim, adj_lists, aggregator,num_sample=10,base_model=None, gcn=False, cuda=False, feature_transform=False
        self.enc1 = Encoder(None, 3, 256, adj_lists, self.agg1)
        self.agg2 = MeanAggregator(lambda raw_features, nodes : self.enc1(raw_features, nodes).t())
        self.enc2 = Encoder(lambda raw_features, nodes : self.enc1(raw_features, nodes).t(), self.enc1.embed_dim, 
                            512, adj_lists, self.agg2)    
        self.agg3 = MeanAggregator(lambda raw_features, nodes : self.enc2(raw_features, nodes).t())
        self.enc3 = Encoder(lambda raw_features, nodes : self.enc2(raw_features, nodes).t(), self.enc2.embed_dim, 
                            512, adj_lists, self.agg3)    
        self.agg4 = MeanAggregator(lambda raw_features, nodes : self.enc3(raw_features, nodes).t())
        self.enc4 = Encoder(lambda raw_features, nodes : self.enc3(raw_features, nodes).t(), self.enc3.embed_dim, 
                            512, adj_lists, self.agg4)    
        #self.agg5 = MeanAggregator(lambda raw_features, nodes : self.enc4(raw_features, nodes).t())
        #self.enc5 = Encoder(lambda raw_features, nodes : self.enc4(raw_features, nodes).t(), self.enc4.embed_dim, 
         #                   256, adj_lists, self.agg5)    
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, self.enc4.embed_dim))
        init.xavier_uniform(self.weight)
        
    def forward(self, raw_features, nodes):
        embeds = self.enc4(raw_features, nodes)
        scores = self.weight.mm(embeds)
        return scores.t()
      

batches_per_file = int(math.ceil(10242.0/batch_size))
model = SupervisedGraphSage(36)
model.cuda()
optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=0, weight_decay=wd)
criterion = nn.CrossEntropyLoss()

def val_during_training():
    model.eval()
    
    total_correct = 0
    for file_idx in range(len(val_files)):
        file = files[file_idx]
        feats = sio.loadmat(file)
        feats = feats['data']
        label = sio.loadmat(file[:-4] + '.label')
        label = label['label']    
        label = np.squeeze(label)
        label = label - 1
        features = nn.Embedding(10242, 3)
        features.weight = nn.Parameter(torch.FloatTensor(feats), requires_grad=False)
        
        current_correct = 0
        for local_batch_idx in range(batches_per_file):
            batch_nodes = [batch_node_idx for batch_node_idx in range(local_batch_idx * batch_size, 
                 (local_batch_idx+1) * batch_size)] if local_batch_idx != (batches_per_file - 1) else [batch_node_idx for batch_node_idx in range(local_batch_idx * batch_size, len(feats))]
            batch_labels = label[np.array(batch_nodes)]
            batch_labels = Variable(torch.LongTensor(batch_labels)).cuda()
            pred = model(features, batch_nodes)
            current_correct += pred.data.max(1)[1].eq(batch_labels.data).long().cpu().sum()
        total_correct += current_correct
        print("validate valdataset during training:{}/{} ACC={}".format(file_idx, len(val_files), current_correct/10242.0))

    val_acc = total_correct / (10242.0 * len(val_files))
    
#    total_correct = 0
#    for file_idx in range(len(files)):
#        file = files[file_idx]
#        feats = sio.loadmat(file)
#        feats = feats['data']
#        label = sio.loadmat(file[:-4] + '.label')
#        label = label['label']    
#        label = np.squeeze(label)
#        label = label - 1
#    
#        features = nn.Embedding(10242, 3)
#        features.weight = nn.Parameter(torch.FloatTensor(feats), requires_grad=False)
#        
#        current_correct = 0
#        for local_batch_idx in range(batches_per_file):
#            batch_nodes = [batch_node_idx for batch_node_idx in range(local_batch_idx * batch_size, 
#                 (local_batch_idx+1) * batch_size)] if local_batch_idx != (batches_per_file - 1) else [batch_node_idx for batch_node_idx in range(local_batch_idx * batch_size, len(feats))]
#            batch_labels = label[np.array(batch_nodes)]
#            batch_labels = Variable(torch.LongTensor(batch_labels)).cuda()
#            pred = model(features, batch_nodes)
#            current_correct += pred.data.max(1)[1].eq(batch_labels.data).long().cpu().sum()
#        total_correct += current_correct
#        print("validate traindataset during training:{}/{} ACC={}".format(file_idx, len(files), current_correct/10242.0))
#
#    train_acc = total_correct / (10242.0 * len(files))
    
    return val_acc


def get_learning_rate(epoch):
    limits = [2, 5, 20]
    lrs = [1, 0.1, 0.01, 0.001]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate

for epoch in range(100):
    lr = get_learning_rate(epoch)
    print("learning rate = {} and batch size = {}".format(lr, batches_per_file))
    for p in optimizer.param_groups:
       # print(p)
        p['lr'] = lr
    for file_idx in range(len(files)):
        file = files[file_idx]
        feats = sio.loadmat(file)
        feats = feats['data']
        label = sio.loadmat(file[:-4] + '.label')
        label = label['label']    
        label = np.squeeze(label)
        label = label - 1
        
        features = nn.Embedding(10242, 3)
        features.weight = nn.Parameter(torch.FloatTensor(feats), requires_grad=False)
        
        for local_batch_idx in range(batches_per_file):
            model.train()
            batch_nodes = [batch_node_idx for batch_node_idx in range(local_batch_idx * batch_size, 
                 (local_batch_idx+1) * batch_size)] if local_batch_idx != (batches_per_file - 1) else [batch_node_idx for batch_node_idx in range(local_batch_idx * batch_size, len(feats))]
            batch_labels = label[np.array(batch_nodes)]
            batch_labels = Variable(torch.LongTensor(batch_labels)).cuda()
            
            optimizer.zero_grad()
            pred = model(features, batch_nodes)
            loss = criterion(pred, batch_labels)
            loss.backward()
            optimizer.step()
           
            correct = pred.data.max(1)[1].eq(batch_labels.data).long().cpu().sum()
            print("[{}:{}:{}] LOSS={:.2} ACC={:.4}".format(epoch, file_idx, 
                    local_batch_idx, loss.data.cpu().numpy()[0], float(correct) / len(batch_labels)))
        
    val_acc = val_during_training()
#    writer.add_scalars('data/Acc', {'train': train_acc, 'val': val_acc},  epoch*len(files)+file_idx)

    torch.save(model.state_dict(), os.path.join("state.pkl"))
