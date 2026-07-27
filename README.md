# Explainable Movie Recommendation using Graph Attention Networks (GAT)
### A Graph Theory Approach — 6th Semester Mini Project

## 1. Project Motivation

Traditional recommender systems assign **fixed, manually chosen weights**
to different features:

```
Genre    = 0.4
Actor    = 0.3
Director = 0.2
Producer = 0.1
```

These weights never change and are not justified by data.

This project instead models movies, genres, actors, directors and
producers as a **heterogeneous graph** and uses a **Graph Attention
Network (GAT)** to *learn* how much attention each neighbor type
deserves — automatically, from graph structure:

```
Genre    -> 0.48   (learned)
Actor    -> 0.31
Director -> 0.17
Producer -> 0.04
```

The learned attention coefficients are extracted, visualized, and
turned into a natural-language explanation by a locally hosted LLM
(Qwen2.5-7B-Instruct), making the whole pipeline **explainable**.

## 2. Graph Theory Concepts Demonstrated

| Concept | Where it appears |
|---|---|
| Heterogeneous graph | Movie / Genre / Actor / Director / Producer node types |
| Directed graph | `movie -> genre`, plus explicit reverse edges |
| Weighted graph | `similar_to` edges weighted by genre-overlap similarity |
| Neighborhood aggregation | GATConv layers aggregating neighbor features |
| Edge importance / attention | Learned `alpha` coefficients per edge |
| Graph visualization | NetworkX + Matplotlib, before/after training |

## 3. Project Structure

```
movie-gat-project/
├── dataset/                # MovieLens download + metadata synthesis
├── preprocessing/          # NetworkX graph construction, PyG conversion
├── models/
│   ├── traditional/        # Manual-weight content-based recommender
│   └── gat/                # GAT model, training, attention extraction, recommender
├── visualization/          # NetworkX/Matplotlib graph & chart rendering
├── llm/                    # Local Qwen2.5-7B-Instruct explanation module
├── ui/                     # Streamlit application
├── utils/                  # Shared helpers
├── requirements.txt
└── README.md
```

## 4. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on PyTorch Geometric**: depending on your OS/CUDA version you
> may need to install `torch` first, then `torch-geometric` following
> the official instructions at https://pytorch-geometric.readthedocs.io

> **Note on the LLM**: `Qwen/Qwen2.5-7B-Instruct` is a 7-billion
> parameter model (~15GB in fp16). A CUDA GPU with at least 16GB VRAM
> is recommended. If unavailable, uncheck "Use local Qwen2.5-7B" in
> the sidebar to fall back to a fast rule-based explanation generator
> that uses the exact same attention data.

## 5. Dataset

We use the **real TMDB 5000 Movie Dataset** (movies + credits, sourced
from The Movie Database). Nothing is randomly generated: **director**,
**top-billed actors**, and **producer** are parsed directly from TMDB's
real crew/cast JSON, and **avg_rating**/**num_ratings** are TMDB's real
aggregated `vote_average` / `vote_count` for each movie — not synthetic
placeholders.

- 4,748 real movies (after dropping entries with no genre/cast data)
- Real genres, real release years
- Real cast lists (top 5 billed actors per movie)
- Real director (from crew job == "Director")
- Real producer (from crew job == "Producer")
- Real TMDB audience rating and vote count

Source files (auto-downloaded by `dataset/download_data.py`, no API key needed):
- https://raw.githubusercontent.com/andandandand/CSV-datasets/master/tmdb_5000_movies.csv
- https://raw.githubusercontent.com/andandandand/CSV-datasets/master/tmdb_5000_credits.csv

(Originally the well-known "TMDB 5000 Movie Dataset" distributed on Kaggle;
these are public GitHub mirrors of the same CSV files.)

Run once manually (optional — the Streamlit UI also does this automatically):

```bash
python dataset/download_data.py
```

## 6. Running the App

```bash
streamlit run ui/streamlit_app.py
```

Then in the sidebar, in order:

1. **Load / Download Dataset**
2. **Build Graph**
3. **Train GAT** (adjust epoch slider as desired, default 100)

Once trained, use the tabs to explore:

- **Graph Overview** — node/edge counts and graph-theory summary
- **Graph Visualization** — full graph (uniform edges) vs. attention-weighted ego-graph
- **Recommendations** — switch between Traditional and GAT methods
- **Attention Explainability** — bar charts comparing manual vs. learned weights, plus LLM explanation
- **Training Diagnostics** — training loss curve and attention coefficient histogram

## 7. Model Details

`models/gat/gat_model.py` implements a 2-layer GAT:

```
GATConv(in_dim -> hidden_dim, heads=4) -> ELU -> Dropout
GATConv(hidden_dim*4 -> out_dim, heads=1) -> Linear
```

Training (`models/gat/train.py`) is **self-supervised**: a BPR-style
pairwise ranking loss encourages embeddings of connected nodes to be
more similar than embeddings of random (unconnected) node pairs. No
manual labels are required — attention patterns emerge purely from
graph structure and node features.

Attention coefficients are extracted via PyTorch Geometric's
`return_attention_weights=True` and aggregated per neighbor-type in
`models/gat/attention_utils.py` to produce the Genre/Actor/Director/
Producer contribution breakdown shown in the UI.

## 8. Comparison Methodology

| | Traditional | GAT |
|---|---|---|
| Weights | Fixed, manual | Learned via attention |
| Graph-aware | No | Yes |
| Explainability | Fixed formula | Data-driven attention breakdown |
| Adaptivity | Same for every movie | Per-movie, per-neighbor adaptive |

## 9. Future Work

- Incorporate real TMDB/OMDb metadata instead of synthesized crew data
- Add a `User -> Movie` edge type and personalize recommendations per user
- Experiment with `GATv2Conv` for more expressive attention
- Add quantitative offline evaluation (Recall@K, NDCG) against held-out ratings
- Fine-tune the local LLM on attention-to-explanation pairs for more consistent explanations
- Deploy with a persistent vector index (FAISS) for larger catalogs

## 10. Screenshots

_Add screenshots here after running the app locally, e.g.:_

- `screenshots/graph_overview.png`
- `screenshots/attention_visualization.png`
- `screenshots/recommendation_comparison.png`
