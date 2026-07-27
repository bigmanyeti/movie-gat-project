"""
Builds a heterogeneous graph from the enriched MovieLens metadata.

Node types : movie, genre, actor, director
             (producer is intentionally NOT modeled -- see note below)
Edge types : (movie, has_genre, genre)
             (movie, has_actor, actor)
             (movie, directed_by, director)
             (movie, similar_to, movie)        -- shared-genre similarity
             (movie, same_franchise, movie)    -- shared root TITLE
                                                   (e.g. Toy Story / Toy
                                                   Story 2 / Toy Story 3),
                                                   weighted higher than
                                                   plain genre similarity
                                                   so title carries more
                                                   influence in the GAT.

Producer nodes/edges have been removed entirely from this graph (the
"traditional" fixed-weight baseline in models/traditional/ has had its
weights renormalized across genre/actor/director accordingly).

The graph is represented both as a NetworkX graph (for visualization)
and as a PyTorch Geometric HeteroData object (for GAT training). To
keep the GAT implementation simple and fast for a semester project,
we also collapse the heterogeneous graph into a single homogeneous
graph with typed node-feature blocks, which GATConv can operate on
directly. Node type is preserved as metadata for visualization and
explanation.
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer

from utils.title_utils import root_title

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed")
GRAPH_CACHE = os.path.join(PROCESSED_DIR, "graph_cache.pkl")


class MovieGraphBuilder:
    """Builds and caches the heterogeneous movie graph."""

    def __init__(self, movies_csv=None):
        self.movies_csv = movies_csv or os.path.join(PROCESSED_DIR, "movies_enriched.csv")
        self.movies_df = None
        self.G = nx.DiGraph()
        self.node_type = {}
        self.node_index = {}
        self.index_node = {}

    def load(self):
        self.movies_df = pd.read_csv(self.movies_csv)
        self.movies_df["genres_list"] = self.movies_df["genres"].apply(
            lambda g: [x for x in str(g).split("|") if x and x != "(no genres listed)"]
        )
        self.movies_df["actors_list"] = self.movies_df["actors"].apply(
            lambda a: str(a).split("|")
        )
        return self.movies_df

    def _add_node(self, key, ntype, **attrs):
        if key not in self.node_index:
            idx = len(self.node_index)
            self.node_index[key] = idx
            self.index_node[idx] = key
            self.node_type[key] = ntype
            self.G.add_node(key, ntype=ntype, **attrs)
        return self.node_index[key]

    def build(self, similarity_top_k=5):
        if self.movies_df is None:
            self.load()

        # Movie nodes
        for _, row in self.movies_df.iterrows():
            mkey = f"movie::{row['movieId']}"
            self._add_node(
                mkey,
                "movie",
                title=row["title"],
                year=int(row["year"]),
                avg_rating=float(row["avg_rating"]),
                num_ratings=float(row["num_ratings"]),
            )

        # Genre, actor, director nodes + edges
        for _, row in self.movies_df.iterrows():
            mkey = f"movie::{row['movieId']}"

            for genre in row["genres_list"]:
                gkey = f"genre::{genre}"
                self._add_node(gkey, "genre", name=genre)
                self.G.add_edge(mkey, gkey, etype="has_genre", weight=1.0)
                self.G.add_edge(gkey, mkey, etype="rev_has_genre", weight=1.0)

            for actor in row["actors_list"]:
                akey = f"actor::{actor}"
                self._add_node(akey, "actor", name=actor)
                self.G.add_edge(mkey, akey, etype="has_actor", weight=1.0)
                self.G.add_edge(akey, mkey, etype="rev_has_actor", weight=1.0)

            dkey = f"director::{row['director']}"
            self._add_node(dkey, "director", name=row["director"])
            self.G.add_edge(mkey, dkey, etype="directed_by", weight=1.0)
            self.G.add_edge(dkey, mkey, etype="rev_directed_by", weight=1.0)

        # Movie -> Movie similarity edges (based on shared genre overlap)
        self._add_similarity_edges(top_k=similarity_top_k)

        # Movie -> Movie franchise edges (based on shared root TITLE) --
        # gives title a direct, strong structural role in the graph, on
        # top of (and weighted higher than) plain genre-overlap similarity.
        self._add_title_franchise_edges()

        return self.G

    def _add_similarity_edges(self, top_k=5):
        """
        Computes movie-movie similarity edges based on shared genres.

        Uses sparse matrix operations instead of a dense N x N similarity
        matrix — at ~10k movies a dense matrix would be ~10681^2 floats
        (~900MB in float64), which is wasteful and slow to build. Since
        each movie only has a handful of genres, the genre membership
        matrix is highly sparse, so a sparse dot product plus per-row
        top-k extraction is both faster and far lighter on memory.
        """
        from scipy import sparse

        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(self.movies_df["genres_list"])
        genre_sparse = sparse.csr_matrix(genre_matrix, dtype=np.float32)
        movie_ids = self.movies_df["movieId"].tolist()

        sim_sparse = genre_sparse @ genre_sparse.T  # sparse dot product, shape (N, N)
        norms = np.asarray(genre_sparse.sum(axis=1)).flatten()
        norms[norms == 0] = 1

        sim_sparse = sim_sparse.tocsr()

        for i, mid in enumerate(movie_ids):
            row_start, row_end = sim_sparse.indptr[i], sim_sparse.indptr[i + 1]
            col_indices = sim_sparse.indices[row_start:row_end]
            col_scores = sim_sparse.data[row_start:row_end] / norms[i]

            if len(col_scores) == 0:
                continue

            # top_k+1 to account for the self-similarity entry, filtered below
            k = min(top_k + 1, len(col_scores))
            top_local = np.argpartition(-col_scores, k - 1)[:k]
            top_local = top_local[np.argsort(-col_scores[top_local])]

            mkey = f"movie::{mid}"
            added = 0
            for local_idx in top_local:
                j = col_indices[local_idx]
                if j == i:
                    continue
                score = float(col_scores[local_idx])
                if score <= 0:
                    continue
                other_mid = movie_ids[j]
                okey = f"movie::{other_mid}"
                self.G.add_edge(mkey, okey, etype="similar_to", weight=score)
                added += 1
                if added >= top_k:
                    break

    def _add_title_franchise_edges(self):
        """
        Connects movies that share the same "root title" (i.e. the same
        franchise once sequel numbering/roman numerals are stripped --
        see utils/title_utils.root_title), for example:

            "Toy Story", "Toy Story 2", "Toy Story 3"  -> all linked

        These edges get a much higher weight (FRANCHISE_EDGE_WEIGHT) than
        the plain genre-overlap `similar_to` edges, and are additionally
        leveraged directly at recommendation time in gat_recommender.py.
        """
        FRANCHISE_EDGE_WEIGHT = 5.0

        groups = defaultdict(list)
        for _, row in self.movies_df.iterrows():
            rt = root_title(row["title"])
            if rt and len(rt) > 2:
                groups[rt].append(row["movieId"])

        for rt, movie_ids in groups.items():
            if len(movie_ids) < 2:
                continue
            for i, mid_a in enumerate(movie_ids):
                for mid_b in movie_ids[i + 1:]:
                    akey, bkey = f"movie::{mid_a}", f"movie::{mid_b}"
                    self.G.add_edge(akey, bkey, etype="same_franchise", weight=FRANCHISE_EDGE_WEIGHT)
                    self.G.add_edge(bkey, akey, etype="same_franchise", weight=FRANCHISE_EDGE_WEIGHT)

    def node_features(self):
        """
        Builds a numeric feature matrix for every node, using simple
        one-hot / hashed encodings appropriate for a graph-theory
        focused mini project (feature richness is intentionally kept
        interpretable rather than maximized).

        12-dimensional feature vector:
            [0:4)  -- node-type one-hot (movie, genre, actor, director)
            [4:12) -- 8-dim hashed identity embedding

        Note on what was intentionally removed from the original 16-dim
        vector: normalized release year, normalized avg rating, and
        log-scaled popularity (num_ratings) are no longer encoded as
        node features -- ranking/recommendation should be driven by
        graph structure (genre/actor/director/title) rather than by a
        movie's release date or how popular/well-rated it already is.
        The `producer` node type/one-hot slot has also been removed
        entirely (see class docstring).

        Title weighting: for MOVIE nodes, the 8-dim hashed identity
        embedding is derived from the movie's franchise ROOT TITLE
        (see utils/title_utils.root_title) rather than its raw title.
        This means "Toy Story", "Toy Story 2", and "Toy Story 3" all
        hash to the exact same identity block, giving title a strong,
        direct influence on the resulting GAT embeddings on top of the
        explicit `same_franchise` graph edges added above. Non-movie
        nodes (genre/actor/director) keep a plain name-based hash.
        """
        n = len(self.node_index)
        feat_dim = 12
        X = np.zeros((n, feat_dim), dtype=np.float32)

        type_onehot = {"movie": 0, "genre": 1, "actor": 2, "director": 3}

        for key, idx in self.node_index.items():
            data = self.G.nodes[key]
            ntype = data["ntype"]
            X[idx, type_onehot[ntype]] = 1.0

            # Hashed identity embedding (8-dim hash bucket), used to break
            # symmetry between different actors/directors/genres, and --
            # for movies -- to encode franchise identity via the root title.
            if ntype == "movie":
                name = root_title(data.get("title", "")) or data.get("title") or key
            else:
                name = data.get("name") or key
            h = abs(hash(name))
            for k in range(8):
                bit = (h >> k) & 1
                X[idx, 4 + k] = float(bit)

        return X

    def save_cache(self):
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(GRAPH_CACHE, "wb") as f:
            pickle.dump(
                {
                    "G": self.G,
                    "node_index": self.node_index,
                    "index_node": self.index_node,
                    "node_type": self.node_type,
                },
                f,
            )

    @classmethod
    def load_cache(cls):
        with open(GRAPH_CACHE, "rb") as f:
            data = pickle.load(f)
        builder = cls()
        builder.G = data["G"]
        builder.node_index = data["node_index"]
        builder.index_node = data["index_node"]
        builder.node_type = data["node_type"]
        return builder

    @staticmethod
    def cache_exists():
        return os.path.exists(GRAPH_CACHE)


if __name__ == "__main__":
    builder = MovieGraphBuilder()
    builder.load()
    G = builder.build()
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    builder.save_cache()