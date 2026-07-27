"""
Trains MovieGAT using a self-supervised link-prediction objective:
for each existing edge (positive) we sample a random non-edge
(negative) and train the model so that the dot product of the
resulting node embeddings is higher for positive pairs than for
negative pairs (standard BPR-style pairwise loss). This requires no
manual labels and lets the attention weights emerge purely from
graph structure + node features, which is exactly the "learned
weights vs manual weights" story this project is built around.

Best-epoch search + retrain
----------------------------
Rather than just training once for a fixed number of epochs and
keeping whatever the final-epoch weights happen to be, `train_gat()`
now does a two-phase process:

    Phase 1 (search):  train a model for up to `max_epochs` epochs,
                        recording the loss at every epoch, and find
                        the epoch with the LOWEST loss (best_epoch).

    Phase 2 (retrain):  re-initialize a FRESH model (same architecture
                        + same random seed) and train it from scratch
                        for EXACTLY `best_epoch` epochs. This retrained
                        model -- not the phase-1 model -- is what gets
                        saved to disk and used for recommendations.

Because both phases use the same seed and the same hyperparameters,
phase 2 deterministically reproduces phase 1's trajectory up through
best_epoch, so the saved model is exactly the lowest-loss point found
during the search, obtained via an explicit from-scratch retrain
rather than by simply checkpointing mid-run.
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

DEFAULT_SEED = 42


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_negative_edges(num_nodes, num_samples, existing_edges_set):
    neg = []
    while len(neg) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in existing_edges_set:
            neg.append((u, v))
    return neg


def _run_training(data, epochs, seed, lr=0.01, hidden_dim=32, out_dim=32, heads=4,
                   dropout=0.3, log_callback=None):
    """
    Runs one full from-scratch training loop for `epochs` epochs with a
    freshly initialized model, seeded deterministically by `seed`.
    Returns (model, embeddings, attention_info, loss_history, elapsed_seconds, num_params).
    """
    _seed_everything(seed)

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
    return model, embeddings, attention_info, loss_history, elapsed_seconds, num_params


def train_gat(data, epochs=100, lr=0.01, hidden_dim=32, out_dim=32, heads=4, dropout=0.3,
              seed=DEFAULT_SEED, log_callback=None, search_log_callback=None):
    """
    Two-phase training entry point used by the app:

      1. SEARCH  -- train for up to `epochs` epochs, tracking loss per
                    epoch, to find the epoch with the lowest loss.
      2. RETRAIN -- re-initialize a fresh model with the same seed and
                    train it from scratch for exactly that many epochs.
                    This retrained model is what gets saved to disk.

    `epochs` is therefore a MAXIMUM / search budget, not necessarily the
    number of epochs the final saved model was actually trained for --
    check `train_stats["best_epoch"]` for the number actually used.

    log_callback, if given, is called during the RETRAIN phase (the run
    whose progress corresponds to the final saved model). Pass
    search_log_callback to also observe progress during the search
    phase (e.g. to show a separate progress bar for it in the UI).
    """
    # ---- Phase 1: search for the best (lowest-loss) epoch ----
    _, _, _, search_loss_history, search_elapsed, num_params = _run_training(
        data, epochs=epochs, seed=seed, lr=lr, hidden_dim=hidden_dim, out_dim=out_dim,
        heads=heads, dropout=dropout, log_callback=search_log_callback,
    )

    best_epoch = int(np.argmin(search_loss_history)) + 1  # 1-indexed epoch COUNT to retrain for
    best_loss = float(search_loss_history[best_epoch - 1])

    # ---- Phase 2: fresh retrain for exactly `best_epoch` epochs ----
    model, embeddings, attention_info, retrain_loss_history, retrain_elapsed, _ = _run_training(
        data, epochs=best_epoch, seed=seed, lr=lr, hidden_dim=hidden_dim, out_dim=out_dim,
        heads=heads, dropout=dropout, log_callback=log_callback,
    )

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
        "elapsed_seconds": search_elapsed + retrain_elapsed,
        "search_elapsed_seconds": search_elapsed,
        "retrain_elapsed_seconds": retrain_elapsed,
        "num_params": num_params,
        "num_nodes": int(data.x.size(0)),
        "num_edges": int(data.edge_index.size(1)),
        "search_epochs": epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "search_loss_history": search_loss_history,
        "loss_history": retrain_loss_history,
        "loss_start": retrain_loss_history[0] if retrain_loss_history else None,
        "loss_end": retrain_loss_history[-1] if retrain_loss_history else None,
    }

    return model, embeddings.detach().numpy(), attention_info, retrain_loss_history, train_stats


def load_trained_embeddings():
    with open(EMB_PATH, "rb") as f:
        return pickle.load(f)


def load_attention():
    with open(ATTN_PATH, "rb") as f:
        return pickle.load(f)


def embeddings_exist():
    return os.path.exists(EMB_PATH) and os.path.exists(ATTN_PATH)