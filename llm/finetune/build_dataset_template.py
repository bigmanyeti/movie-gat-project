"""
Generates a REAL fine-tuning dataset (JSONL) from your already-trained
GAT model's actual attention outputs, instead of hand-written mock
examples — this is how you scale train_qlora.py's 3-example demo up
to 5,000+ real examples.

Strategy:
    1. For every movie in the graph, get its top-K GAT recommendations
       (already implemented in models/gat/gat_recommender.py) and its
       real attention breakdown (models/gat/attention_utils.py).
    2. Use the BASE (non-fine-tuned) Qwen model, or the rule-based
       fallback in llm/explainer.py, to generate an initial explanation
       for each (source, recommended, metrics) triple.
    3. Optionally have a human spot-check / lightly edit a sample of
       these for quality before training on them (recommended for a
       few hundred at minimum) — this is standard practice: use a
       larger/better model or human review to bootstrap SFT data for
       a smaller fine-tune target.
    4. Save everything as JSONL with the same schema train_qlora.py's
       `RAW_EXAMPLES` uses, so you can swap the loader in one line:

           from datasets import load_dataset
           train_dataset = load_dataset(
               "json", data_files="llm/finetune/data/explanations.jsonl", split="train"
           )
           # then map through format_example()/apply_chat_template as in train_qlora.py

Run (after you've already built the graph and trained the GAT once
via the Streamlit app, so the cached files exist):
    python llm/finetune/build_dataset_template.py --num-movies 2000 --top-k 3
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from preprocessing.build_graph import MovieGraphBuilder
from models.gat.train import load_trained_embeddings, load_attention
from models.gat.gat_recommender import GATRecommender
from models.gat.attention_utils import movie_attention_breakdown
from llm.explainer import explain_recommendation  # rule-based fallback works without GPU

OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "explanations.jsonl")


def main(num_movies, top_k, use_llm):
    if not MovieGraphBuilder.cache_exists():
        raise SystemExit("No cached graph found — build the graph via the Streamlit app first.")

    builder = MovieGraphBuilder.load_cache()
    embeddings = load_trained_embeddings()
    attn_data = load_attention()
    recommender = GATRecommender(builder, embeddings)

    movie_keys = [k for k, v in builder.node_type.items() if v == "movie"]
    movie_ids = [int(k.split("::")[1]) for k in movie_keys][:num_movies]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    written = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for movie_id in movie_ids:
            mkey = f"movie::{movie_id}"
            source_title = builder.G.nodes[mkey].get("title", str(movie_id))

            recs = recommender.recommend(movie_id, top_k=top_k)
            breakdown = movie_attention_breakdown(mkey, builder, attn_data)
            metrics = breakdown.get("normalized_contribution", {})
            if not metrics:
                continue

            for rec in recs:
                explanation = explain_recommendation(
                    source_title, rec["title"], breakdown, method="GAT", use_llm=use_llm,
                )
                record = {
                    "source_movie": source_title,
                    "recommended_movie": rec["title"],
                    "metrics": metrics,
                    "explanation": explanation,
                }
                f.write(json.dumps(record) + "\n")
                written += 1

    print(f"Wrote {written} training examples to {OUT_PATH}")
    print("Recommended next step: open this file and spot-check / lightly "
          "edit a sample (at least a few hundred rows) for quality before "
          "using it as SFT training data — bootstrapped explanations from "
          "a rule-based or base-model generator are a good starting point, "
          "not a substitute for review.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-movies", type=int, default=2000,
                         help="How many movies to generate examples for (each produces --top-k rows).")
    parser.add_argument("--top-k", type=int, default=3,
                         help="Recommendations per movie -> total rows = num_movies * top_k.")
    parser.add_argument("--use-llm", action="store_true",
                         help="Use the base Qwen model to generate initial explanations "
                              "(slower, needs GPU) instead of the fast rule-based fallback.")
    args = parser.parse_args()
    main(args.num_movies, args.top_k, args.use_llm)
