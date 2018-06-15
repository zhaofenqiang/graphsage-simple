import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
import glob
import math
import os
from sklearn.metrics import f1_score
from collections import defaultdict
import scipy.io as sio 

from graphsage.encoders import Encoder
from graphsage.parcellation_aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

learning_rate = 0.1
wd = 0.0001
batch_size = 4
train_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90'
files = sorted(glob.glob(os.path.join(train_dataset_path, '*.mat'))) 

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes):
        super(SupervisedGraphSage, self).__init__()

        self.agg1 = MeanAggregator(features=None, cuda=True)
        #features, feature_dim, embed_dim, adj_lists, aggregator,num_sample=10,base_model=None, gcn=False, cuda=False, feature_transform=False
        self.enc1 = Encoder(features=None, 3, 32, adj_mat, self.agg1, cuda=True)
        self.agg2 = MeanAggregator(lambda nodes : self.enc1(nodes).t(), cuda=True)
        self.enc2 = Encoder(lambda nodes : self.enc1(nodes).t(), self.enc1.embed_dim, 
                            64, adj_mat, self.agg2, base_model=self.enc1, cuda=True)    
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, self.enc2.embed_dim))
        init.xavier_uniform(self.weight)
        self.xent = nn.CrossEntropyLoss()
        

    def forward(self, raw_features, nodes):
        embeds = self.enc2(raw_features, nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def get_learning_rate(epoch):
    limits = [10, 20, 40]
    lrs = [1, 0.1, 0.01, 0.001]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate



adj_mat = sio.loadmat('/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/adj_mat.mat')
adj_mat = adj_mat['adj_mat']
optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0, weight_decay=wd)
batches_per_file = int(math.ceil(10242.0/batch_size))




graphsage = SupervisedGraphSage(36)

for epoch in range(100):
    lr = get_learning_rate(epoch)
    print("learning rate = {} and batch size = {}".format(lr, batches_per_file))
    for p in optimizer.param_groups:
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
            batch_nodes = [batch_node_idx for batch_node_idx in range(local_batch_idx * batch_size, 
                 (local_batch_idx+1) * batch_size)] if local_batch_idx != (batches_per_file - 1) else [batch_node_idx for batch_node_idx in range(local_batch_idx * batch_size, len(img))]
            batch_labels = label[np.array(batch_nodes)]
            
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(batch_labels)))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            print batch, loss.data[0]
       
    
    
    
    













def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    #features, feature_dim, embed_dim, adj_lists, aggregator,num_sample=10,base_model=None, gcn=False, cuda=False, feature_transform=False
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:10]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.data[0]

    val_output = graphsage.forward(val) 
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.data[0]

    val_output = graphsage.forward(val) 
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)

if __name__ == "__main__":
    run_cora()

