"""
Visualization utilities for the movie knowledge graph.

Provides:
    - draw_full_graph        : whole graph, uniform edge width (pre-training view)
    - draw_movie_neighborhood: ego-graph around one movie, edge width = attention
    - attention_bar_chart    : bar chart of Genre/Actor/Director contribution
    - training_loss_plot     : line chart of GAT training loss
    - attention_histogram    : histogram of all learned attention coefficients
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

TYPE_COLORS = {
    "movie": "#4C9AFF",
    "genre": "#57D9A3",
    "actor": "#FFAB00",
    "director": "#FF5630",
}


def _node_colors(G, nodes):
    return [TYPE_COLORS.get(G.nodes[n].get("ntype", "movie"), "#999999") for n in nodes]


def draw_full_graph(G, max_nodes=150, seed=42):
    if G.number_of_nodes() > max_nodes:
        # Sample a connected-ish subgraph for readability
        nodes = list(G.nodes())[:max_nodes]
        sub = G.subgraph(nodes)
    else:
        sub = G

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    pos = nx.spring_layout(sub, seed=seed, k=0.4)
    node_colors = _node_colors(sub, sub.nodes())

    nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=120, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(sub, pos, width=0.5, alpha=0.3, edge_color="#888888", ax=ax, arrows=False)

    ax.set_title("Movie Knowledge Graph (uniform edge width — before training)",
                 color="white", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_movie_neighborhood(G, movie_key, attention_breakdown, radius=1, seed=7):
    """
    Draws the ego-graph around a movie, with edge thickness and color
    scaled by the learned attention weight of that neighbor's type.
    """
    if movie_key not in G:
        raise ValueError(f"{movie_key} not found in graph")

    ego = nx.ego_graph(G, movie_key, radius=radius, undirected=True)

    top_neighbors = attention_breakdown.get("top_neighbors", {})
    score_lookup = {}
    for ntype, edges in top_neighbors.items():
        for node_key, score in edges:
            score_lookup[node_key] = score

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    pos = nx.spring_layout(ego, seed=seed, k=0.6)

    node_colors = _node_colors(ego, ego.nodes())
    node_sizes = [900 if n == movie_key else 300 for n in ego.nodes()]

    nx.draw_networkx_nodes(ego, pos, node_color=node_colors, node_size=node_sizes,
                            ax=ax, edgecolors="white", linewidths=0.5)

    labels = {}
    for n in ego.nodes():
        data = ego.nodes[n]
        labels[n] = data.get("title") or data.get("name") or n
    nx.draw_networkx_labels(ego, pos, labels=labels, font_size=7, font_color="white", ax=ax)

    edges = list(ego.edges())
    widths = []
    colors = []
    for u, v in edges:
        neighbor = v if u == movie_key else u
        score = score_lookup.get(neighbor, 0.05)
        widths.append(1 + score * 15)
        colors.append(plt.cm.plasma(min(score * 3, 1.0)))

    nx.draw_networkx_edges(ego, pos, width=widths, edge_color=colors, ax=ax, arrows=False, alpha=0.85)

    ax.set_title(f"Attention-weighted neighborhood of: {G.nodes[movie_key].get('title', movie_key)}",
                 color="white", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def attention_bar_chart(normalized_contribution, title="Learned Attention Contribution"):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    items = sorted(normalized_contribution.items(), key=lambda x: -x[1])
    labels = [k.capitalize() for k, _ in items]
    values = [v for _, v in items]
    colors = [TYPE_COLORS.get(k, "#999999") for k, _ in items]

    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", color="white", fontsize=9)

    ax.set_ylim(0, max(values + [0.1]) * 1.25)
    ax.set_ylabel("Contribution", color="white")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.tight_layout()
    return fig


def comparison_bar_chart(rows, title="Manual Weights vs GAT-Learned Weights"):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    labels = [r["type"].capitalize() for r in rows]
    manual = [r["manual_weight"] for r in rows]
    learned = [r["learned_weight"] for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, manual, width, label="Manual (Traditional)", color="#888888")
    ax.bar(x + width / 2, learned, width, label="Learned (GAT)", color="#4C9AFF")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white")
    ax.set_ylabel("Weight", color="white")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0E1117", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.tight_layout()
    return fig


def training_loss_plot(loss_history):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    ax.plot(loss_history, color="#57D9A3", linewidth=2)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.set_title("GAT Training Loss", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.tight_layout()
    return fig


def attention_histogram(alpha_values):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")
    ax.hist(alpha_values, bins=30, color="#FFAB00", alpha=0.85)
    ax.set_xlabel("Attention coefficient", color="white")
    ax.set_ylabel("Frequency", color="white")
    ax.set_title("Distribution of Learned Attention Coefficients", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.tight_layout()
    return fig