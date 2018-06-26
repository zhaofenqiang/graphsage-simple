import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, feature_dim, 
            embed_dim, adj_lists, aggregator, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator

        self.embed_dim = embed_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(feature_dim, embed_dim))
        init.xavier_uniform(self.weight)
 
        self.bn = nn.BatchNorm1d(embed_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, raw_features, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """

        neigh_feats = self.aggregator.forward(raw_features, nodes, [self.adj_lists[node] for node in nodes])

        x = neigh_feats.mm(self.weight)
        x = self.bn(x)
        x = self.relu(x)
        return x
