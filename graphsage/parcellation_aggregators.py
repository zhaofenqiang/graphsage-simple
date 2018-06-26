import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, feature_dim, embed_dim): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.FloatTensor(7 * feature_dim, embed_dim))
        self.bn = nn.BatchNorm1d(embed_dim)
        self.relu = nn.LeakyReLU()
        init.xavier_uniform(self.weight)
 

    def forward(self, raw_features, nodes, to_neighs):
        """
        nodes --- list of nodes in a batch
        like  [491, 988, 1446, 1611, 2217, 824, 494, 799, 1001, 847]
        
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        like [{182, 468, 2318, 2377},
         {1515},
         {948, 1282},
         {100, 163, 1153, 1379, 1380},
         {1007, 2282},
         {1215, 1621, 1707},
         {427, 747},
         {1122, 1699, 1706, 2656},
         {1354},
         {710, 848, 967, 1032, 1402, 1468}]
        
        num_sample --- number of neighbors to sample. No sampling if None.
        """

        for i, samp_neigh in enumerate(to_neighs):
            
            if self.features:
                neigh_mat = self.features(raw_features, torch.LongTensor(list(samp_neigh)))
            else:
                neigh_mat = raw_features(torch.LongTensor(list(samp_neigh))).cuda()
            
            if len(neigh_mat) == 6:
                neigh_mat = torch.cat((neigh_mat, torch.mean(neigh_mat, 0, True)), 0)
            neigh_mat = neigh_mat.contiguous()
            neigh_mat = neigh_mat.view(1, 7 * self.feat_dim)
            if i == 0:
                mat = neigh_mat
            else:
                mat = torch.cat((mat,neigh_mat),0)
                

        x = mat.mm(self.weight)
        x = self.bn(x)
        x = self.relu(x)
        return x
