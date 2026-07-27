"""
Trains MovieGAT using a self-supervised link-prediction objective:
for each existing edge (positive) we sample a random non-edge
(negative) and train the model so that the dot product of the
resulting node embeddings is higher for positive pairs than for
negative pairs (standard BPR-style pairwise loss). This requires no
manual labels and lets the attention weights emerge purely from
graph structure + node features, which is exactly the "learned
weights vs manual weights" story this project is built around.
"""

import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.gat.gat_model import MovieGAT

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "processed")
MODEL_PATH = os.path.join(MODEL_DIR, "gat_model.pt")
EMB_PATH = os.path.join(MODEL_DIR, "gat_embeddings.pkl")
ATTN_PATH = os.path.join(MODEL_DIR, "gat_attention.pkl")


def sample_negative_edges(num_nodes, num_samples, existing_edges_set):
    neg = []
    while len(neg) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in existing_edges_set:
            neg.append((u, v))
    return neg


def train_gat(data, epochs=100, lr=0.01, hidden_dim=32, out_dim=32, heads=4, dropout=0.3,
              log_callback=None):
    num_nodes = data.x.size(0)
    edge_index = data.edge_index

    existing_edges_set = set(
        (int(u), int(v)) for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist())
    )
    pos_edges = list(existing_edges_set)

    model = MovieGAT(
        in_dim=data.x.size(1), hidden_dim=hidden_dim, out_dim=out_dim,
        heads=heads, dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    num_params = sum(p.numel() for p in model.parameters())

    loss_history = []
    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings, attention_info = model(data.x, edge_index)

        batch_size = min(2048, len(pos_edges))
        batch_pos = random.sample(pos_edges, batch_size)
        batch_neg = sample_negative_edges(num_nodes, batch_size, existing_edges_set)

        pos_u = torch.tensor([p[0] for p in batch_pos])
        pos_v = torch.tensor([p[1] for p in batch_pos])
        neg_u = torch.tensor([n[0] for n in batch_neg])
        neg_v = torch.tensor([n[1] for n in batch_neg])

        # Normalize before scoring: recommendations are made using cosine
        # similarity at inference time, so training must optimize the same
        # angular objective. Without this, raw dot products can grow in a
        # shared direction (embeddings collapse into a narrow cone) — the
        # loss goes down, but every pair ends up with near-identical
        # cosine similarity, making the resulting recommendations
        # meaningless (everything looks equally "similar").
        norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
        pos_score = (norm_embeddings[pos_u] * norm_embeddings[pos_v]).sum(dim=-1)
        neg_score = (norm_embeddings[neg_u] * norm_embeddings[neg_v]).sum(dim=-1)

        loss = F.softplus((neg_score - pos_score) * 10.0).mean()
        reg = 1e-4 * embeddings.norm(dim=-1).mean()
        total_loss = loss + reg

        total_loss.backward()
        optimizer.step()

        loss_history.append(float(total_loss.item()))
        if log_callback:
            log_callback(epoch, float(total_loss.item()))
        elif epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss {total_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings, attention_info = model(data.x, edge_index)

    elapsed_seconds = time.time() - start_time

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    with open(EMB_PATH, "wb") as f:
        pickle.dump(embeddings.detach().numpy(), f)

    attn_serializable = {
        "layer2_edge_index": attention_info["layer2"]["edge_index"].detach().numpy(),
        "layer2_alpha": attention_info["layer2"]["alpha"].detach().numpy(),
    }
    with open(ATTN_PATH, "wb") as f:
        pickle.dump(attn_serializable, f)

    train_stats = {
        "elapsed_seconds": elapsed_seconds,
        "num_params": num_params,
        "num_nodes": num_nodes,
        "num_edges": edge_index.size(1),
        "loss_start": loss_history[0] if loss_history else None,
        "loss_end": loss_history[-1] if loss_history else None,
    }

    return model, embeddings.detach().numpy(), attention_info, loss_history, train_stats


def load_trained_embeddings():
    with open(EMB_PATH, "rb") as f:
        return pickle.load(f)


def load_attention():
    with open(ATTN_PATH, "rb") as f:
        return pickle.load(f)


def embeddings_exist():
    return os.path.exists(EMB_PATH) and os.path.exists(ATTN_PATH)
