"""
Produces recommendations from trained GAT node embeddings using
cosine similarity.

Two modes:
    recommend(movie_id)         -> movies similar to a movie you've watched
                                    (target vector = that movie's own embedding)
    recommend_by_genre(genre)   -> top movies for a chosen genre
                                    (target vector = the GENRE NODE's own
                                    learned embedding -- genres are real
                                    nodes in the graph with their own
                                    representation, shaped by attention
                                    over every movie connected to them)
"""

import numpy as np


class GATRecommender:
    def __init__(self, builder, embeddings):
        self.builder = builder
        self.embeddings = embeddings

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

    def _rank_movies_by_similarity(self, target_idx, top_k, exclude_idx=None):
        node_index = self.builder.node_index
        index_node = self.builder.index_node
        node_type = self.builder.node_type

        target_vec = self.embeddings[target_idx]
        target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)

        movie_indices = [
            idx for key, idx in node_index.items()
            if node_type[key] == "movie" and idx != exclude_idx
        ]
        movie_vecs = self.embeddings[movie_indices]
        norms = np.linalg.norm(movie_vecs, axis=1, keepdims=True) + 1e-8
        movie_vecs_norm = movie_vecs / norms

        sims = movie_vecs_norm @ target_norm

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
        """Movies similar to a specific movie you've already watched."""
        node_index = self.builder.node_index
        mkey = self._movie_key(movie_id)
        if mkey not in node_index:
            return []
        target_idx = node_index[mkey]
        return self._rank_movies_by_similarity(target_idx, top_k, exclude_idx=target_idx)

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

        target_vec = self.embeddings[target_idx]
        target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)

        index_node = self.builder.index_node
        movie_indices = list(valid_movie_indices)
        movie_vecs = self.embeddings[movie_indices]
        norms = np.linalg.norm(movie_vecs, axis=1, keepdims=True) + 1e-8
        movie_vecs_norm = movie_vecs / norms
        sims = movie_vecs_norm @ target_norm

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
