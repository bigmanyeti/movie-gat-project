"""
Converts the NetworkX heterogeneous graph produced by build_graph.py
into a homogeneous PyTorch Geometric `Data` object suitable for
GATConv training, while retaining node-type metadata for later use
in visualization and explanation.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from preprocessing.build_graph import MovieGraphBuilder


def graph_to_pyg_data(builder: MovieGraphBuilder):
    G = builder.G
    node_index = builder.node_index

    X = builder.node_features()
    x = torch.tensor(X, dtype=torch.float32)

    src, dst, edge_type, edge_weight = [], [], [], []
    etype_map = {}

    for u, v, data in G.edges(data=True):
        et = data.get("etype", "unknown")
        if et not in etype_map:
            etype_map[et] = len(etype_map)
        src.append(node_index[u])
        dst.append(node_index[v])
        edge_type.append(etype_map[et])
        edge_weight.append(data.get("weight", 1.0))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index)
    data.edge_type = edge_type_t
    data.edge_weight = edge_weight_t
    data.etype_map = etype_map

    movie_mask = np.array(
        [builder.node_type[builder.index_node[i]] == "movie" for i in range(len(node_index))]
    )
    data.movie_mask = torch.tensor(movie_mask, dtype=torch.bool)

    return data
