"""
Explainable Movie Recommendation using Graph Attention Networks
A Graph Theory mini-project — Streamlit UI

Run with:
    streamlit run ui/streamlit_app.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from dataset.download_data import build_dataset
from preprocessing.build_graph import MovieGraphBuilder
from preprocessing.to_pyg import graph_to_pyg_data
from models.gat.train import train_gat, load_attention
from models.gat.attention_utils import movie_attention_breakdown, compare_with_manual_weights
from models.gat.gat_recommender import GATRecommender
from models.traditional.traditional_recommender import TraditionalRecommender
from visualization import graph_viz
from llm.explainer import (
    explain_recommendation, finetuned_adapter_available,
    load_finetune_status, load_finetune_loss_history,
)
from utils.helpers import inject_dark_theme, movie_title_lookup

st.set_page_config(page_title="Explainable GAT Movie Recommender", layout="wide", page_icon="🎬")
inject_dark_theme()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULTS = {
    "movies_df": None,
    "builder": None,
    "pyg_data": None,
    "traditional_rec": None,
    "embeddings": None,
    "attn_data": None,
    "loss_history": None,
    "trained_epochs": None,
    "train_stats": None,
    # results are keyed as {(movie_id, method): [..]} so switching movie or
    # method NEVER silently shows results computed for a different combo.
    "results_cache": {},
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.title("🎬 Explainable Movie Recommendation using Graph Attention Networks")
st.caption("A Graph Theory Mini Project — comparing manually weighted recommenders "
           "against attention-based Graph Attention Networks (GAT).")

# ---------------------------------------------------------------------------
# Sidebar — strict step-by-step pipeline
# ---------------------------------------------------------------------------
dataset_ready = st.session_state.movies_df is not None
graph_ready = st.session_state.builder is not None

with st.sidebar:
    st.header("Pipeline")
    st.caption("Steps unlock in order: Load Dataset → Build Graph → Train GAT.")

    if st.button("1. Load / Download Dataset", use_container_width=True):
        with st.spinner("Preparing MovieLens dataset..."):
            st.session_state.movies_df = build_dataset()
        # A fresh dataset invalidates everything downstream.
        st.session_state.builder = None
        st.session_state.pyg_data = None
        st.session_state.traditional_rec = None
        st.session_state.embeddings = None
        st.session_state.attn_data = None
        st.session_state.loss_history = None
        st.session_state.trained_epochs = None
        st.session_state.results_cache = {}
        st.success(f"Loaded {len(st.session_state.movies_df)} movies.")
        st.rerun()

    if st.button("2. Build Graph", use_container_width=True, disabled=not dataset_ready):
        with st.spinner("Building heterogeneous graph..."):
            builder = MovieGraphBuilder()
            builder.load()
            builder.build()
            builder.save_cache()
            st.session_state.builder = builder
            st.session_state.pyg_data = graph_to_pyg_data(builder)
            st.session_state.traditional_rec = TraditionalRecommender(st.session_state.movies_df)
        # A rebuilt graph invalidates any embeddings/attention trained on the
        # OLD graph — without this, switching movies would show attention
        # weights that no longer correspond to the current graph structure.
        st.session_state.embeddings = None
        st.session_state.attn_data = None
        st.session_state.loss_history = None
        st.session_state.trained_epochs = None
        st.session_state.results_cache = {}
        st.success(
            f"Graph built: {builder.G.number_of_nodes()} nodes, "
            f"{builder.G.number_of_edges()} edges."
        )
        st.rerun()
    if not dataset_ready:
        st.caption("🔒 Locked until dataset is loaded.")

    epochs = st.slider("Training epochs", 20, 300, 100, step=10, disabled=not graph_ready)

    if st.button("3. Train GAT", use_container_width=True, disabled=not graph_ready):
        progress = st.progress(0.0)
        status = st.empty()

        def log_cb(epoch, loss):
            progress.progress(min((epoch + 1) / epochs, 1.0))
            status.text(f"Epoch {epoch + 1}/{epochs} — loss {loss:.4f}")

        with st.spinner("Training Graph Attention Network..."):
            model, embeddings, attn_info, loss_history, train_stats = train_gat(
                st.session_state.pyg_data, epochs=epochs, log_callback=log_cb
            )
            st.session_state.embeddings = embeddings
            st.session_state.attn_data = load_attention()
            st.session_state.loss_history = loss_history
            st.session_state.trained_epochs = epochs
            st.session_state.train_stats = train_stats
        # New training run -> any cached GAT recommendations are stale.
        st.session_state.results_cache = {
            k: v for k, v in st.session_state.results_cache.items() if k[1] != "GAT"
        }
        st.success(
            f"GAT training complete in {train_stats['elapsed_seconds']:.1f}s "
            f"({train_stats['num_params']:,} parameters, "
            f"{train_stats['num_nodes']:,} nodes, {train_stats['num_edges']:,} edges). "
            f"Loss {train_stats['loss_start']:.3f} → {train_stats['loss_end']:.3f}."
        )
        st.rerun()
    if not graph_ready:
        st.caption("🔒 Locked until graph is built.")

    st.divider()
    st.subheader("Status")
    st.markdown(f"- Dataset loaded: {'✅' if dataset_ready else '❌'}")
    st.markdown(f"- Graph built: {'✅' if graph_ready else '❌'}")
    st.markdown("- Traditional method: ✅ ready (no training needed, only requires graph)"
                if graph_ready else "- Traditional method: ❌ needs graph built first")
    if st.session_state.trained_epochs is not None:
        st.markdown(f"- GAT method: ✅ trained ({st.session_state.trained_epochs} epochs)")
    else:
        st.markdown("- GAT method: ❌ not trained yet")

    st.divider()
    use_llm = st.checkbox("Use local Qwen2.5-7B for explanations", value=False,
                           help="Requires the model to be downloaded and a capable GPU. "
                                "If unchecked, a fast rule-based explanation is used instead.")

    adapter_ready = finetuned_adapter_available()
    use_finetuned = False
    if use_llm:
        if adapter_ready:
            use_finetuned = st.checkbox(
                "Use fine-tuned LoRA adapter (trained on GAT metrics)", value=True,
                help="Uses the QLoRA adapter trained by llm/finetune/train_qlora.py "
                     "instead of the base Qwen model.",
            )
        else:
            st.caption("ℹ️ No fine-tuned adapter found yet — using base Qwen model. "
                       "Run `llm/finetune/train_qlora.py` to train one; it will "
                       "appear here automatically once training completes.")

# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------
if not dataset_ready:
    st.info("👈 Click **1. Load / Download Dataset** in the sidebar to get started.")
    st.stop()

movies_df = st.session_state.movies_df
title_lookup = movie_title_lookup(movies_df)

col_select, col_topk = st.columns([3, 1])
with col_select:
    movie_id = st.selectbox(
        "Select a movie",
        options=movies_df["movieId"].tolist(),
        format_func=lambda mid: title_lookup.get(mid, str(mid)),
    )
with col_topk:
    top_k = st.number_input("Top-K", min_value=1, max_value=15, value=5)

tabs = st.tabs([
    "📊 Graph Overview", "🕸️ Graph Visualization", "🎯 Recommendations",
    "🔍 Attention Explainability", "📈 Training Diagnostics", "🧬 LLM Fine-Tuning",
])

# ---------------- Tab 1: Graph Overview ----------------
with tabs[0]:
    st.subheader("Graph Theory Summary")
    if not graph_ready:
        st.warning("Build the graph first using the sidebar.")
    else:
        G = st.session_state.builder.G
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Nodes", G.number_of_nodes())
        c2.metric("Total Edges", G.number_of_edges())
        c3.metric("Movie Nodes", sum(1 for n, d in G.nodes(data=True) if d["ntype"] == "movie"))
        c4.metric("Avg Degree", f"{sum(dict(G.degree()).values()) / G.number_of_nodes():.1f}")

        st.markdown("""
        **Node types:** Movie, Genre, Actor, Director, Producer
        **Edge types:** `has_genre`, `has_actor`, `directed_by`, `produced_by`, `similar_to`
        (plus reverse edges to keep the graph traversable in both directions).

        This is a **heterogeneous, weighted, directed graph** — the foundation for
        applying Graph Attention Networks, since GATs learn how much attention
        (importance) to place on each neighboring node during aggregation.
        """)

# ---------------- Tab 2: Graph Visualization ----------------
with tabs[1]:
    st.subheader("Graph Visualization")
    if not graph_ready:
        st.warning("Build the graph first using the sidebar.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Before training** — all edges equal width")
            fig = graph_viz.draw_full_graph(st.session_state.builder.G)
            st.pyplot(fig, use_container_width=True)
        with right:
            st.markdown("**After training** — edge width = learned attention")
            if st.session_state.attn_data is None:
                st.info("Attention data not available yet — click **3. Train GAT** in the sidebar.")
            else:
                st.caption(f"Showing attention from the GAT trained for "
                           f"{st.session_state.trained_epochs} epochs on the current graph. "
                           f"The model is trained once on the whole graph — switching movies "
                           f"here does not retrain anything, it just looks up that movie's "
                           f"node in the already-trained model.")
                mkey = f"movie::{movie_id}"
                breakdown = movie_attention_breakdown(mkey, st.session_state.builder, st.session_state.attn_data)
                fig2 = graph_viz.draw_movie_neighborhood(st.session_state.builder.G, mkey, breakdown)
                st.pyplot(fig2, use_container_width=True)

# ---------------- Tab 3: Recommendations ----------------
with tabs[2]:
    st.subheader("Recommendations")
    if not graph_ready:
        st.warning("Build the graph first using the sidebar.")
    else:
        gat_ready = st.session_state.embeddings is not None

        rec_mode = st.radio(
            "Recommend based on:",
            ["A movie I've watched", "A genre"],
            horizontal=True,
            help="'A genre' skips picking a specific movie -- just choose a genre "
                 "and get the top real movies in it.",
        )

        method_options = ["Traditional"] + (["GAT"] if gat_ready else [])
        method = st.radio(
            "Recommendation method",
            method_options,
            horizontal=True,
            help="GAT only appears here once it has been trained in the sidebar.",
        )
        if not gat_ready:
            st.caption("ℹ️ GAT option will appear here once you train it in the sidebar. "
                       "Traditional needs no training and is ready now.")

        if st.session_state.traditional_rec is None:
            st.session_state.traditional_rec = TraditionalRecommender(movies_df)

        if rec_mode == "A genre":
            genre_list = st.session_state.traditional_rec.available_genres()
            selected_genre = st.selectbox("Select a genre", genre_list)
            current_key = ("genre", selected_genre, method)
        else:
            current_key = ("movie", movie_id, method)

        if st.button("🎯 Generate Recommendations", type="primary"):
            if method == "Traditional":
                rec = st.session_state.traditional_rec
                if rec_mode == "A genre":
                    results = rec.recommend_by_genre(selected_genre, top_k=top_k)
                else:
                    results = rec.recommend(movie_id, top_k=top_k)
            else:
                rec = GATRecommender(st.session_state.builder, st.session_state.embeddings)
                if rec_mode == "A genre":
                    results = rec.recommend_by_genre(selected_genre, top_k=top_k)
                else:
                    results = rec.recommend(movie_id, top_k=top_k)
            st.session_state.results_cache[current_key] = results

        results = st.session_state.results_cache.get(current_key)
        if results:
            if rec_mode == "A genre":
                if method == "Traditional":
                    st.markdown(
                        f"Top **{selected_genre}** movies, ranked by IMDB-style weighted "
                        f"rating (real avg_rating + real vote count, corrected for small "
                        f"sample sizes)."
                    )
                    st.dataframe(
                        [{"Title": r["title"], "Weighted score": round(r["score"], 3),
                          "Raw rating": r["avg_rating"], "Votes": int(r["num_ratings"])}
                         for r in results],
                        use_container_width=True,
                    )
                else:
                    st.markdown(
                        f"Top **{selected_genre}** movies (verified genre membership), "
                        f"ranked by cosine similarity to the GAT's learned embedding for "
                        f"the '{selected_genre}' genre node."
                    )
                    st.dataframe(
                        [{"Title": r["title"], "Score": round(r["score"], 4)} for r in results],
                        use_container_width=True,
                    )
            else:
                if method == "Traditional":
                    st.markdown("Computed with **fixed manual weights**: Genre 0.4, Actor 0.3, Director 0.2, Producer 0.1")
                else:
                    st.markdown("Computed with **GAT-learned embeddings** and cosine similarity.")
                st.dataframe(
                    [{"Title": r["title"], "Score": round(r["score"], 4)} for r in results],
                    use_container_width=True,
                )
        else:
            st.info("Click **Generate Recommendations** above to compute results for "
                    "this selection + method combination.")

# ---------------- Tab 4: Attention Explainability ----------------
with tabs[3]:
    st.subheader("Why was this recommended?")
    if not graph_ready:
        st.warning("Build the graph first using the sidebar.")
    else:
        gat_ready = st.session_state.embeddings is not None
        method_options = ["Traditional"] + (["GAT"] if gat_ready else [])
        explain_method = st.radio("Method to explain", method_options, horizontal=True, key="explain_method")

        explain_mode = st.radio(
            "Recommendations from:", ["A movie I've watched", "A genre"],
            horizontal=True, key="explain_mode",
        )

        if st.session_state.traditional_rec is None:
            st.session_state.traditional_rec = TraditionalRecommender(movies_df)

        if explain_mode == "A genre":
            genre_list = st.session_state.traditional_rec.available_genres()
            explain_genre = st.selectbox("Genre", genre_list, key="explain_genre_select")
            current_key = ("genre", explain_genre, explain_method)
        else:
            current_key = ("movie", movie_id, explain_method)

        results = st.session_state.results_cache.get(current_key)

        if not results:
            st.info(f"No {explain_method} recommendations generated yet for this selection. "
                    "Go to the Recommendations tab, match this exact mode/genre-or-movie/method, "
                    "and click Generate Recommendations first.")
        else:
            rec_options = {r["title"]: r for r in results}
            chosen_title = st.selectbox("Choose a recommended movie to explain", list(rec_options.keys()))
            chosen = rec_options[chosen_title]

            if explain_mode == "A genre":
                # No single "source movie" in genre mode -- instead show
                # why THIS movie itself ranks well (its own real signals).
                st.markdown(f"Explaining why **{chosen_title}** ranks highly for "
                            f"**{explain_genre}** using **{explain_method}**.")

                if explain_method == "GAT":
                    mkey = f"movie::{chosen['movieId']}"
                    breakdown = movie_attention_breakdown(mkey, st.session_state.builder, st.session_state.attn_data)
                    contribution = breakdown.get("normalized_contribution", {})
                    fig = graph_viz.attention_bar_chart(
                        contribution, title=f"{chosen_title}'s own attention breakdown"
                    )
                    st.pyplot(fig, use_container_width=True)
                    source_label = f"the {explain_genre} genre"
                else:
                    st.markdown(
                        f"**Real rating:** {chosen.get('avg_rating', 'N/A')} / 10 "
                        f"from **{int(chosen.get('num_ratings', 0)):,}** votes -- "
                        f"**weighted score:** {chosen['score']:.3f} "
                        f"(corrects for vote-count reliability, see Recommendations tab)."
                    )
                    breakdown = {"normalized_contribution": {
                        "rating": chosen.get("avg_rating", 0) / 10,
                        "vote_confidence": min(chosen.get("num_ratings", 0) / 10000, 1.0),
                    }}
                    source_label = f"the {explain_genre} genre"

                if st.button("🧠 Generate LLM Explanation", key="genre_explain_btn"):
                    with st.spinner("Generating explanation..."):
                        explanation = explain_recommendation(
                            source_label, chosen_title, breakdown,
                            method=explain_method, use_llm=use_llm, use_finetuned=use_finetuned,
                        )
                    variant_label = "fine-tuned adapter" if use_finetuned else ("base Qwen model" if use_llm else "rule-based")
                    st.caption(f"Generated using: {variant_label}")
                    st.markdown(f"> {explanation}")

            else:
                st.markdown(f"Explaining recommendations for **{title_lookup.get(movie_id)}** "
                            f"using **{explain_method}**.")

                if explain_method == "GAT":
                    mkey = f"movie::{movie_id}"
                    breakdown = movie_attention_breakdown(mkey, st.session_state.builder, st.session_state.attn_data)
                    contribution = breakdown.get("normalized_contribution", {})

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        fig = graph_viz.attention_bar_chart(contribution)
                        st.pyplot(fig, use_container_width=True)
                    with c2:
                        rows = compare_with_manual_weights(contribution)
                        fig2 = graph_viz.comparison_bar_chart(rows)
                        st.pyplot(fig2, use_container_width=True)

                    if st.button("🧠 Generate LLM Explanation", key="gat_explain_btn"):
                        with st.spinner("Generating explanation..."):
                            explanation = explain_recommendation(
                                title_lookup.get(movie_id, str(movie_id)),
                                chosen["title"],
                                breakdown,
                                method="GAT",
                                use_llm=use_llm,
                                use_finetuned=use_finetuned,
                            )
                        variant_label = "fine-tuned adapter" if use_finetuned else ("base Qwen model" if use_llm else "rule-based")
                        st.caption(f"Generated using: {variant_label}")
                        st.markdown(f"> {explanation}")
                else:
                    source = st.session_state.traditional_rec
                    # Use source.movies_df (not the raw movies_df) — it has the
                    # genres_set / actors_set columns that _similarity() needs.
                    target_row = source.movies_df[source.movies_df["movieId"] == movie_id].iloc[0]
                    chosen_row = source.movies_df[source.movies_df["movieId"] == chosen["movieId"]].iloc[0]
                    score, breakdown_manual = source._similarity(target_row, chosen_row)
                    fig = graph_viz.attention_bar_chart(
                        dict(breakdown_manual), title="Manual Weight Contribution"
                    )
                    st.pyplot(fig, use_container_width=True)

                    if st.button("🧠 Generate LLM Explanation", key="trad_explain_btn"):
                        fake_breakdown = {"normalized_contribution": breakdown_manual}
                        with st.spinner("Generating explanation..."):
                            explanation = explain_recommendation(
                                title_lookup.get(movie_id, str(movie_id)),
                                chosen["title"],
                                fake_breakdown,
                                method="Traditional",
                                use_llm=use_llm,
                                use_finetuned=use_finetuned,
                            )
                        variant_label = "fine-tuned adapter" if use_finetuned else ("base Qwen model" if use_llm else "rule-based")
                        st.caption(f"Generated using: {variant_label}")
                        st.markdown(f"> {explanation}")

# ---------------- Tab 5: Training Diagnostics ----------------
with tabs[4]:
    st.subheader("Training Diagnostics")
    if st.session_state.loss_history is None:
        st.info("Train the GAT to see diagnostics.")
    else:
        st.caption(f"From the most recent training run ({st.session_state.trained_epochs} epochs).")

        stats = st.session_state.train_stats
        if stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Training time", f"{stats['elapsed_seconds']:.1f}s")
            c2.metric("Model parameters", f"{stats['num_params']:,}")
            c3.metric("Loss (start)", f"{stats['loss_start']:.3f}")
            c4.metric("Loss (end)", f"{stats['loss_end']:.3f}")
            improvement = (stats['loss_start'] - stats['loss_end']) / max(abs(stats['loss_start']), 1e-8) * 100
            st.caption(f"Loss dropped {improvement:.1f}% over training — confirms the model "
                       f"is actually learning, not just running instantly with no effect. "
                       f"Small graph size ({stats['num_nodes']:,} nodes, {stats['num_edges']:,} edges) "
                       f"and a lightweight 2-layer GAT ({stats['num_params']:,} parameters) is why "
                       f"this completes in seconds rather than minutes.")
        fig = graph_viz.training_loss_plot(st.session_state.loss_history)
        st.pyplot(fig, use_container_width=True)

        alpha = st.session_state.attn_data["layer2_alpha"].reshape(-1)
        fig2 = graph_viz.attention_histogram(alpha)
        st.pyplot(fig2, use_container_width=True)

# ---------------- Tab 6: LLM Fine-Tuning ----------------
with tabs[5]:
    st.subheader("LLM Fine-Tuning Diagnostics")
    st.caption("Tracks the separate QLoRA fine-tuning job (llm/finetune/train_qlora.py) "
               "that teaches Qwen2.5-7B to write explanations natively from GAT metrics, "
               "as opposed to the base model's prompt-only explanations used elsewhere in this app.")

    status = load_finetune_status()
    if status is None:
        st.info(
            "No completed fine-tuning run found yet. This is a separate, much heavier "
            "job than GAT training — run it from a terminal (not from this app):\n\n"
            "```\npython llm/finetune/build_dataset_template.py --num-movies 2000 --top-k 3\n"
            "python llm/finetune/train_qlora.py\n```\n\n"
            "Once it completes, this tab will automatically show its real training "
            "time, loss curve, and example count — refresh the page after training finishes."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training examples", f"{status['num_examples']:,}")
        c2.metric("Epochs", status['num_epochs'])
        mins = status['elapsed_seconds'] / 60
        c3.metric("Training time", f"{mins:.1f} min")
        if status.get('first_loss') is not None and status.get('final_loss') is not None:
            drop = (status['first_loss'] - status['final_loss']) / max(abs(status['first_loss']), 1e-8) * 100
            c4.metric("Loss drop", f"{drop:.1f}%")
        st.caption(f"Adapter trained at: {status['trained_at']} — saved to `{status['adapter_dir']}`")

        loss_history = load_finetune_loss_history()
        if loss_history:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [h["step"] for h in loss_history]
            losses = [h["loss"] for h in loss_history]

            fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0E1117")
            ax.set_facecolor("#0E1117")
            ax.plot(steps, losses, color="#FFAB00", linewidth=2, marker="o", markersize=3)
            ax.set_xlabel("Training step", color="white")
            ax.set_ylabel("Loss", color="white")
            ax.set_title("QLoRA Fine-Tuning Loss (real logged values from trainer)", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#444444")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            st.caption(
                f"This many-hour-scale run ({mins:.1f} minutes with {status['num_examples']:,} "
                f"examples in this case — scale examples/epochs up for a genuinely overnight run) "
                f"is fundamentally different from the ~2-minute GAT training: it's updating LoRA "
                f"adapter weights layered onto a 7-billion-parameter language model, versus the "
                f"GAT's ~7,680 parameters on a small graph."
            )
        else:
            st.warning("Status file found but no loss history logged — check that training "
                       "completed at least one logging step.")

        st.divider()
        st.markdown("**Try it**: once trained, toggle *'Use fine-tuned LoRA adapter'* in the "
                    "sidebar (under 'Use local Qwen2.5-7B for explanations') and generate an "
                    "explanation in the Attention Explainability tab to compare it against the "
                    "base model's phrasing.")
