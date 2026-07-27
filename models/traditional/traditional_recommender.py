"""
Traditional content-based recommender using MANUALLY assigned,
fixed feature weights. This serves as the baseline the GAT
recommender is compared against.

Similarity is computed as a weighted sum of:
    - Genre overlap (Jaccard)
    - Actor overlap (Jaccard)
    - Director match (binary)
    - Producer match (binary)

with fixed weights:
    genre    = 0.4
    actor    = 0.3
    director = 0.2
    producer = 0.1
"""

import pandas as pd


DEFAULT_WEIGHTS = {
    "genre": 0.4,
    "actor": 0.3,
    "director": 0.2,
    "producer": 0.1,
}


class TraditionalRecommender:
    def __init__(self, movies_df: pd.DataFrame, weights=None):
        self.movies_df = movies_df.copy()
        self.weights = weights or DEFAULT_WEIGHTS
        self.movies_df["genres_set"] = self.movies_df["genres"].apply(
            lambda g: set(x for x in str(g).split("|") if x and x != "(no genres listed)")
        )
        self.movies_df["actors_set"] = self.movies_df["actors"].apply(
            lambda a: set(str(a).split("|"))
        )

    @staticmethod
    def _jaccard(a: set, b: set):
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def _similarity(self, row_a, row_b):
        genre_sim = self._jaccard(row_a["genres_set"], row_b["genres_set"])
        actor_sim = self._jaccard(row_a["actors_set"], row_b["actors_set"])
        director_sim = 1.0 if row_a["director"] == row_b["director"] else 0.0
        producer_sim = 1.0 if row_a["producer"] == row_b["producer"] else 0.0

        score = (
            self.weights["genre"] * genre_sim
            + self.weights["actor"] * actor_sim
            + self.weights["director"] * director_sim
            + self.weights["producer"] * producer_sim
        )
        breakdown = {
            "genre": self.weights["genre"] * genre_sim,
            "actor": self.weights["actor"] * actor_sim,
            "director": self.weights["director"] * director_sim,
            "producer": self.weights["producer"] * producer_sim,
        }
        return score, breakdown

    def recommend(self, movie_id, top_k=5):
        target_rows = self.movies_df[self.movies_df["movieId"] == movie_id]
        if target_rows.empty:
            return []
        target = target_rows.iloc[0]

        results = []
        for _, row in self.movies_df.iterrows():
            if row["movieId"] == movie_id:
                continue
            score, breakdown = self._similarity(target, row)
            results.append({
                "movieId": row["movieId"],
                "title": row["title"],
                "score": score,
                "breakdown": breakdown,
            })

        results.sort(key=lambda r: -r["score"])
        return results[:top_k]

    def available_genres(self):
        """Real genre list pulled from the actual dataset."""
        all_genres = set()
        for genre_set in self.movies_df["genres_set"]:
            all_genres |= genre_set
        return sorted(all_genres)

    def recommend_by_genre(self, genre_name, top_k=5, min_votes_percentile=0.60):
        """
        Top movies within a genre, ranked by IMDB's weighted-rating
        formula rather than raw avg_rating -- this is real statistics,
        not a random or arbitrary sort: a movie with a 9.0 average from
        3 voters should NOT outrank an 8.0 average from 5,000 voters,
        and the weighted rating formula corrects for exactly that.

            WR = (v / (v + m)) * R + (m / (v + m)) * C

        where R = the movie's own avg_rating, v = its num_ratings,
        C = the mean avg_rating across ALL movies in this genre, and
        m = a minimum-votes threshold (here, the given percentile of
        vote counts within the genre) below which a movie's own rating
        gets pulled more heavily toward the genre average.
        """
        subset = self.movies_df[self.movies_df["genres_set"].apply(lambda s: genre_name in s)]
        if subset.empty:
            return []

        C = subset["avg_rating"].mean()
        m = subset["num_ratings"].quantile(min_votes_percentile)

        def weighted_rating(row):
            v = row["num_ratings"]
            R = row["avg_rating"]
            return (v / (v + m)) * R + (m / (v + m)) * C

        subset = subset.copy()
        subset["weighted_rating"] = subset.apply(weighted_rating, axis=1)
        subset = subset.sort_values("weighted_rating", ascending=False)

        results = []
        for _, row in subset.head(top_k).iterrows():
            results.append({
                "movieId": row["movieId"],
                "title": row["title"],
                "score": float(row["weighted_rating"]),
                "avg_rating": float(row["avg_rating"]),
                "num_ratings": float(row["num_ratings"]),
            })
        return results
