"""
Aggregates raw per-edge GAT attention coefficients into a per-neighbor
-type contribution breakdown (Genre / Actor / Director / Producer /
Similar Movie) for a given movie node. This is the core "explainable"
artifact of the project: instead of manually fixed weights, we show
what the model actually learned to pay attention to.
"""

import numpy as np


def movie_attention_breakdown(movie_key, builder, attn_data, layer_heads_mean=True):
    """
    Returns a dict: {neighbor_type: aggregated_attention_score} for all
    edges pointing INTO movie_key in layer 2 of the GAT (i.e. edges
    that were aggregated to form the movie's final embedding).
    """
    node_index = builder.node_index
    index_node = builder.index_node
    node_type = builder.node_type

    if movie_key not in node_index:
        return {}

    target_idx = node_index[movie_key]

    edge_index = attn_data["layer2_edge_index"]  # shape [2, E]
    alpha = attn_data["layer2_alpha"]  # shape [E, heads] or [E, 1]

    if alpha.ndim == 2 and layer_heads_mean:
        alpha_mean = alpha.mean(axis=1)
    else:
        alpha_mean = alpha.reshape(-1)

    dst_nodes = edge_index[1]
    mask = dst_nodes == target_idx
    src_for_target = edge_index[0][mask]
    alpha_for_target = alpha_mean[mask]

    type_scores = {}
    type_edges = {}
    for src_idx, score in zip(src_for_target, alpha_for_target):
        src_key = index_node[int(src_idx)]
        stype = node_type[src_key]
        type_scores[stype] = type_scores.get(stype, 0.0) + float(score)
        type_edges.setdefault(stype, []).append((src_key, float(score)))

    total = sum(type_scores.values())
    if total > 0:
        normalized = {k: v / total for k, v in type_scores.items()}
    else:
        normalized = type_scores

    top_neighbors = {}
    for stype, edges in type_edges.items():
        edges_sorted = sorted(edges, key=lambda e: -e[1])
        top_neighbors[stype] = edges_sorted[:5]

    return {
        "normalized_contribution": normalized,
        "raw_scores": type_scores,
        "top_neighbors": top_neighbors,
    }


def compare_with_manual_weights(learned, manual=None):
    """
    Side-by-side comparison table data between traditional manually
    assigned weights and the GAT-learned attention distribution.
    """
    manual = manual or {"genre": 0.4, "actor": 0.3, "director": 0.2, "producer": 0.1}
    rows = []
    for ntype in ["genre", "actor", "director", "producer"]:
        rows.append({
            "type": ntype,
            "manual_weight": manual.get(ntype, 0.0),
            "learned_weight": learned.get(ntype, 0.0),
        })
    return rows
