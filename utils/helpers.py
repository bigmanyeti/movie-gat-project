"""Small shared helper functions used across the app."""

import streamlit as st


def inject_dark_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        section[data-testid="stSidebar"] {
            background-color: #131720;
        }
        .metric-card {
            background-color: #1A1F2B;
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid #2A2F3B;
        }
        h1, h2, h3 {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def movie_title_lookup(movies_df):
    return dict(zip(movies_df["movieId"], movies_df["title"]))
