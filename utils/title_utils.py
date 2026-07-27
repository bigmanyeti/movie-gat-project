"""
Shared title-normalization utilities used to give a movie's TITLE a
much stronger role in both graph construction (preprocessing/build_graph.py)
and recommendation-time scoring (models/gat/gat_recommender.py).

Core idea: strip sequel/franchise markers ("2", "3", "II", "Part II",
"- Reloaded", etc.) off a title to get its "root title". Movies that
share a root title are the SAME franchise (e.g. "Toy Story",
"Toy Story 2", "Toy Story 3" -> root title "toy story"). This lets us:

    1. Give franchise siblings an (almost) identical node feature
       vector in the GAT graph (see build_graph.py's node_features()),
       so the model starts from a strong shared signal instead of
       purely random-looking hash noise.
    2. Add explicit high-weight `same_franchise` edges directly
       connecting franchise siblings in the graph, so message passing
       pulls their learned embeddings together.
    3. Apply a title-similarity boost + a hard "must include" rule at
       recommendation time, so a query like "Toy Story" reliably
       surfaces "Toy Story 2" / "Toy Story 3" even if the purely
       learned embedding similarity would rank them lower.
"""

import re
from difflib import SequenceMatcher

# Trailing sequel markers: "2", "3", "II", "III", "Part 2", "- Part II", etc.
_TRAILING_NUMERAL = re.compile(r"\s*[:\-]?\s*(part\s+)?(\d+)\s*$", re.IGNORECASE)
_TRAILING_ROMAN = re.compile(
    r"\s*[:\-]?\s*(part\s+)?(ii|iii|iv|v|vi|vii|viii|ix|x)\s*$", re.IGNORECASE
)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_MULTI_SPACE = re.compile(r"\s+")


def root_title(title):
    """
    Normalizes a movie title down to its franchise "root", stripping
    sequel numbering/subtitles and punctuation/case differences.

    "Toy Story"    -> "toy story"
    "Toy Story 2"  -> "toy story"
    "Toy Story 3"  -> "toy story"
    "The Dark Knight Rises" -> "the dark knight rises" (unchanged --
        no trailing sequel marker, so it stays its own root; it is
        NOT force-merged with "The Dark Knight" since that would be a
        false positive for a large fraction of ordinary titles).
    """
    if not title:
        return ""
    t = str(title).strip().lower()
    t = _TRAILING_ROMAN.sub("", t)
    t = _TRAILING_NUMERAL.sub("", t)
    t = _NON_ALNUM.sub(" ", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    return t


def is_same_franchise(title_a, title_b):
    """True if both titles reduce to the same non-trivial root title."""
    ra, rb = root_title(title_a), root_title(title_b)
    return bool(ra) and len(ra) > 2 and ra == rb


def title_similarity(title_a, title_b):
    """
    Continuous title-similarity score in [0, 1] used to weight titles
    heavily at recommendation time. Exact franchise match (same root
    title) scores 1.0; otherwise falls back to fuzzy character-level
    similarity on the raw titles.
    """
    if is_same_franchise(title_a, title_b):
        return 1.0
    return SequenceMatcher(None, str(title_a).lower(), str(title_b).lower()).ratio()