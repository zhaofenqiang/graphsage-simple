import torch
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
    def __init__(self, features, embed_dim): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, 2 * self.feat_dim))
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
        samp_neighs = to_neighs
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.features:
            embed_matrix = self.features(raw_features, torch.LongTensor(unique_nodes_list))
        else:
            embed_matrix = raw_features(torch.LongTensor(unique_nodes_list)).cuda()
            
        to_feats = mask.mm(embed_matrix)
        return to_feats
