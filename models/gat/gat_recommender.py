"""
Produces recommendations from trained GAT node embeddings using
cosine similarity, with TITLE given deliberately increased weight in
the final ranking (see TITLE_WEIGHT below and utils/title_utils.py).

Two modes:
    recommend(movie_id)         -> movies similar to a movie you've watched
                                    (target vector = that movie's own embedding)
    recommend_by_genre(genre)   -> top movies for a chosen genre
                                    (target vector = the GENRE NODE's own
                                    learned embedding -- genres are real
                                    nodes in the graph with their own
                                    representation, shaped by attention
                                    over every movie connected to them)

Title weighting
----------------
On top of the graph-structural `same_franchise` edges added in
preprocessing/build_graph.py (which already pull franchise-sibling
embeddings together during GAT training), `recommend()` re-ranks
candidates using a HYBRID score:

    final_score = (1 - TITLE_WEIGHT) * gat_cosine_similarity
                  + TITLE_WEIGHT * title_similarity

and additionally GUARANTEES that any true franchise sibling (same
root title, e.g. "Toy Story" <-> "Toy Story 2" <-> "Toy Story 3") is
included in the results, sorted to the front by GAT similarity among
themselves. This means querying "Toy Story" is guaranteed to surface
"Toy Story 2" / "Toy Story 3" whenever they exist in the dataset,
rather than leaving that to chance.
"""

import numpy as np

from utils.title_utils import is_same_franchise, title_similarity

# How much weight title similarity gets relative to pure GAT cosine
# similarity when ranking `recommend()` results. 0.5 means title and
# learned graph similarity are each given equal say, with an explicit
# "must include" guarantee for exact franchise matches on top of that.
TITLE_WEIGHT = 0.5


class GATRecommender:
    def __init__(self, builder, embeddings, title_weight=TITLE_WEIGHT):
        self.builder = builder
        self.embeddings = embeddings
        self.title_weight = title_weight

    def _movie_key(self, movie_id):
        return f"movie::{movie_id}"

    def _genre_key(self, genre_name):
        return f"genre::{genre_name}"

    def available_genres(self):
        """Real genre list pulled from the graph's actual genre nodes."""
        return sorted(
            name.split("::", 1)[1] for name, ntype in self.builder.node_type.items()
            if ntype == "genre"
        )

    def _cosine_scores(self, target_idx, movie_indices):
        target_vec = self.embeddings[target_idx]
        target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)

        movie_vecs = self.embeddings[movie_indices]
        norms = np.linalg.norm(movie_vecs, axis=1, keepdims=True) + 1e-8
        movie_vecs_norm = movie_vecs / norms

        return movie_vecs_norm @ target_norm

    def _rank_movies_by_similarity(self, target_idx, top_k, exclude_idx=None):
        """Plain GAT-cosine ranking (no title boost) -- used for genre mode
        and as the underlying signal fed into recommend()'s hybrid score."""
        node_index = self.builder.node_index
        index_node = self.builder.index_node
        node_type = self.builder.node_type

        movie_indices = [
            idx for key, idx in node_index.items()
            if node_type[key] == "movie" and idx != exclude_idx
        ]
        sims = self._cosine_scores(target_idx, movie_indices)

        order = np.argsort(-sims)[:top_k]
        results = []
        for o in order:
            idx = movie_indices[o]
            key = index_node[idx]
            movie_id_result = int(key.split("::")[1])
            title = self.builder.G.nodes[key].get("title", key)
            results.append({
                "movieId": movie_id_result,
                "title": title,
                "score": float(sims[o]),
            })
        return results

    def recommend(self, movie_id, top_k=5):
        """
        Movies similar to a specific movie you've already watched.

        Ranking is a hybrid of GAT-learned cosine similarity and title
        similarity (see module docstring / TITLE_WEIGHT), with a hard
        guarantee that any other movie sharing the same franchise root
        title as the source movie is included in the results.
        """
        node_index = self.builder.node_index
        index_node = self.builder.index_node
        node_type = self.builder.node_type

        mkey = self._movie_key(movie_id)
        if mkey not in node_index:
            return []
        target_idx = node_index[mkey]
        source_title = self.builder.G.nodes[mkey].get("title", mkey)

        movie_indices = [
            idx for key, idx in node_index.items()
            if node_type[key] == "movie" and idx != target_idx
        ]
        if not movie_indices:
            return []

        cos_sims = self._cosine_scores(target_idx, movie_indices)
        # Cosine similarity is in [-1, 1]; rescale to [0, 1] so it combines
        # sensibly with title_similarity (also in [0, 1]).
        cos_sims_01 = (cos_sims + 1.0) / 2.0

        candidates = []
        franchise_siblings = []
        for local_i, idx in enumerate(movie_indices):
            key = index_node[idx]
            title = self.builder.G.nodes[key].get("title", key)
            t_sim = title_similarity(source_title, title)
            hybrid = (1 - self.title_weight) * cos_sims_01[local_i] + self.title_weight * t_sim
            entry = {
                "movieId": int(key.split("::")[1]),
                "title": title,
                "score": hybrid,
                "gat_score": float(cos_sims[local_i]),
            }
            candidates.append(entry)
            if is_same_franchise(source_title, title):
                franchise_siblings.append(entry)

        # Guarantee: franchise siblings always appear, ranked among
        # themselves by pure GAT similarity, followed by the remaining
        # top candidates (by hybrid score) filling out the rest of top_k.
        franchise_siblings.sort(key=lambda r: -r["gat_score"])
        franchise_ids = {r["movieId"] for r in franchise_siblings}

        remaining = [c for c in candidates if c["movieId"] not in franchise_ids]
        remaining.sort(key=lambda r: -r["score"])

        ordered = franchise_siblings + remaining
        results = ordered[:top_k]
        for r in results:
            r.pop("gat_score", None)
            r["score"] = float(r["score"])
        return results

    def recommend_by_genre(self, genre_name, top_k=5):
        """Top movies for a chosen genre.

        Ranking uses cosine similarity to the GENRE NODE's own learned
        embedding -- reflecting what the GAT's attention mechanism
        actually learned. However, similarity alone can occasionally
        surface a movie connected to the genre only indirectly (e.g.
        via a shared actor/director 2 hops away) rather than one that
        truly belongs to that genre. To keep results correct, we first
        filter to movies that ACTUALLY have this genre in their real
        data (ground truth), then use GAT similarity only to rank
        within that already-correct set.
        """
        node_index = self.builder.node_index
        gkey = self._genre_key(genre_name)
        if gkey not in node_index:
            return []
        target_idx = node_index[gkey]

        # Ground-truth filter: only movies genuinely connected to this
        # genre node via a real has_genre edge.
        valid_movie_indices = set()
        for u, v, data in self.builder.G.edges(data=True):
            if data.get("etype") == "has_genre" and v == gkey:
                valid_movie_indices.add(node_index[u])

        if not valid_movie_indices:
            return []

        index_node = self.builder.index_node
        movie_indices = list(valid_movie_indices)
        sims = self._cosine_scores(target_idx, movie_indices)

        order = np.argsort(-sims)[:top_k]
        results = []
        for o in order:
            idx = movie_indices[o]
            key = index_node[idx]
            movie_id_result = int(key.split("::")[1])
            title = self.builder.G.nodes[key].get("title", key)
            results.append({
                "movieId": movie_id_result,
                "title": title,
                "score": float(sims[o]),
            })
        return results