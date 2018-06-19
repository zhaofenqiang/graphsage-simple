import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator

        self.embed_dim = embed_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, 2 * self.feat_dim))
        init.xavier_uniform(self.weight)
        self.relu = nn.LeakyReLU()

    def forward(self, raw_features, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """

        neigh_feats = self.aggregator.forward(raw_features, nodes, [self.adj_lists[node] for node in nodes])

        if self.features:      
            self_feats = self.features(raw_features, torch.LongTensor(nodes))
        else:
            self_feats = raw_features(torch.LongTensor(nodes)).cuda()
                
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = self.relu(self.weight.mm(combined.t()))
        return combined
