"""
Graph Attention Network (GAT) for learning movie embeddings and,
crucially, interpretable attention coefficients over each movie's
heterogeneous neighborhood (genre / actor / director / producer /
similar-movie).

This directly implements the architecture from:
    Velickovic, Cucurull, Casanova, Romero, Lio, Bengio.
    "Graph Attention Networks." ICLR 2018. (arXiv:1710.10903)

Mapping from the paper's notation to this implementation (see the
paper's Section 2.1, Equations 1-6):

    h_i, h_j        -> node feature vectors (this module's `x` rows)
    W                -> GATConv's internal linear transform (per head)
    e_ij = a(Wh_i, Wh_j)
                     -> Eq. 1: raw (unnormalized) attention score between
                        node i and neighbor j, computed via a single-layer
                        feedforward network + LeakyReLU (negative slope
                        0.2, matching the paper) applied to the
                        concatenation [Wh_i || Wh_j] (Eq. 3)
    alpha_ij = softmax_j(e_ij)
                     -> Eq. 2: normalizes each node's neighbor scores to
                        sum to 1 -- these alpha values are exactly what
                        we extract via return_attention_weights=True and
                        later aggregate per neighbor-type (genre/actor/
                        director/producer) for explainability
    h_i' = sigma(sum_j alpha_ij * W h_j)
                     -> Eq. 4: each node's new representation is a
                        weighted (by alpha) sum over its neighborhood
    Multi-head attention (Eq. 5, hidden layers): K independent attention
        mechanisms are computed and their outputs CONCATENATED
        (`concat=True`, PyG's default) -> this is gat1 below (heads=4)
    Multi-head attention (Eq. 6, final/output layer): outputs are
        AVERAGED instead of concatenated, with the nonlinearity applied
        after averaging -> this is gat2 below (heads=1 via concat=False,
        i.e. no concatenation needed since there is effectively one
        averaged output)

Architecture used here (2 layers, matching the paper's transductive
citation-network setup in spirit, adapted for a movie/genre/actor/
director/producer heterogeneous graph):

    Layer 1 (Eq. 5): GATConv(in_dim -> hidden_dim, heads=4, concat=True)
                      -> ELU -> Dropout
    Layer 2 (Eq. 6): GATConv(hidden_dim*heads -> out_dim, heads=1,
                      concat=False i.e. averaged) -> Linear projection

We request `return_attention_weights=True` from both layers so that we
can extract, per edge, the learned attention coefficient alpha_ij.
These are later aggregated per-neighbor-type to produce the
Genre/Actor/Director/Producer contribution breakdown used for
explainability (models/gat/attention_utils.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MovieGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=32, heads=4,
                 dropout=0.3, attn_negative_slope=0.2):
        """
        Args:
            in_dim: input node feature dimension (F in the paper)
            hidden_dim: per-head hidden dimension for layer 1 (F' in Eq. 5)
            out_dim: final embedding dimension after layer 2 + projection
            heads: number of attention heads K for layer 1 (Eq. 5).
                   Layer 2 always uses a single averaged head (Eq. 6).
            dropout: applied to (a) layer inputs and (b) the normalized
                     attention coefficients alpha_ij themselves -- the
                     paper found dropping attention coefficients
                     specifically (not just features) important for
                     regularization on small graphs (Section 3.3)
            attn_negative_slope: LeakyReLU negative slope used inside the
                     attention mechanism a() in Eq. 1/3 -- the paper uses
                     0.2, which we keep as the default here
        """
        super().__init__()
        self.dropout = dropout

        # Layer 1 -- Eq. 5: K=heads independent attention mechanisms,
        # outputs concatenated (concat=True is GATConv's default).
        self.gat1 = GATConv(
            in_dim, hidden_dim, heads=heads,
            dropout=dropout, negative_slope=attn_negative_slope,
        )

        # Layer 2 -- Eq. 6: outputs averaged (concat=False) rather than
        # concatenated, since this is the final representation layer.
        self.gat2 = GATConv(
            hidden_dim * heads, out_dim, heads=1, concat=False,
            dropout=dropout, negative_slope=attn_negative_slope,
        )

        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index):
        # Feature dropout before each layer, as in the paper's setup.
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_1, alpha_1) = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)  # Eq. 5's sigma nonlinearity

        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_2, alpha_2) = self.gat2(x, edge_index, return_attention_weights=True)
        # Eq. 6 delays the final nonlinearity until after averaging heads;
        # here the "final nonlinearity" is left to out_proj + downstream
        # cosine-similarity scoring rather than a softmax/sigmoid, since
        # this is a representation-learning task, not classification.

        embeddings = self.out_proj(x)

        attention_info = {
            "layer1": {"edge_index": edge_index_1, "alpha": alpha_1},
            "layer2": {"edge_index": edge_index_2, "alpha": alpha_2},
        }
        return embeddings, attention_info
