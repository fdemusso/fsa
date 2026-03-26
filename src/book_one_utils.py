"""
utils.py
========
Utility functions for the Rotten Tomatoes recommendation system.

Covers:
  - Dataset download and I/O
  - Review score normalisation
  - Feature engineering (tokenisation, MLB vectorisation, sentence embeddings)
  - Persistence (save / load processed assets)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import re

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import kagglehub
from IPython.display import display
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer


# ===========================================================================
# CONFIGURATION
# ===========================================================================

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"

REVIEWS_PATH = os.path.join(RAW_DIR, "rotten_tomatoes_critic_reviews.csv")
MOVIES_PATH  = os.path.join(RAW_DIR, "rotten_tomatoes_movies.csv")

# Mapping from letter grades to a [0, 1] numeric score
LETTER_GRADE_MAP: dict[str, float] = {
    "A+": 1.0,  "A": 0.95, "A-": 0.90,
    "B+": 0.85, "B": 0.80, "B-": 0.75,
    "C+": 0.65, "C": 0.60, "C-": 0.55,
    "D+": 0.45, "D": 0.40, "D-": 0.35,
    "F":  0.10,
}

# Pre-compiled regexes for score parsing (fraction and percentage formats)
_RE_FRACTION   = re.compile(r"^(\d+[.,]?\d*)\s*/\s*(\d+[.,]?\d*)$")
_RE_PERCENTAGE = re.compile(r"^(\d+[.,]?\d*)\s*%$")

# Sentence-transformer model used for content-based embeddings.
# 'all-MiniLM-L6-v2' offers the best speed / accuracy trade-off (384 dims).
_EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# ===========================================================================
# DATASET ACQUISITION
# ===========================================================================

def download_dataset() -> None:
    """
    Download the Rotten Tomatoes dataset from Kaggle if it is not
    already present on disk.
    """
    if os.path.isfile(REVIEWS_PATH) and os.path.isfile(MOVIES_PATH):
        print("Dataset already downloaded — skipping.")
        return

    print("Dataset not found locally. Downloading from Kaggle …")
    dataset_path = kagglehub.dataset_download(
        "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset",
        output_dir=RAW_DIR,
    )
    print(f"Dataset saved to: {dataset_path}")


def create_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the raw CSV files into DataFrames.

    Returns
    -------
    critic_df : pd.DataFrame
        Raw critic reviews.
    movies_df : pd.DataFrame
        Raw movie metadata.
    """
    movies_df = pd.read_csv(MOVIES_PATH)
    critic_df = pd.read_csv(REVIEWS_PATH)
    return critic_df, movies_df


def print_preview(critic_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
    """Print the first 5 rows of each raw DataFrame for quick inspection."""
    print("=== CRITIC REVIEWS (first 5 rows) ===")
    display(critic_df.head())
    print("\n" + "=" * 45 + "\n")
    print("=== MOVIES METADATA (first 5 rows) ===")
    display(movies_df.head())


# ===========================================================================
# SCORE NORMALISATION
# ===========================================================================

def parse_score(score_str) -> float:
    """
    Convert a raw review score string to a normalised float in [0, 1].

    Supported formats
    -----------------
    - Letter grades : ``A+``, ``B-``, ``F``, …
    - Percentages   : ``85%``
    - Fractions     : ``3.5/5``, ``7/10``, …
    - Plain numbers : ``8`` (assumed base-10 if ≤ 10)

    Returns ``np.nan`` when the format is unrecognised.
    """
    if pd.isna(score_str):
        return np.nan

    # Normalise: strip whitespace, upper-case, replace comma decimals
    s = str(score_str).strip().upper().replace(",", ".")

    # 1. Letter grades
    if s in LETTER_GRADE_MAP:
        return LETTER_GRADE_MAP[s]

    # 2. Percentages  (e.g. "85%")
    m = _RE_PERCENTAGE.match(s)
    if m:
        return min(float(m.group(1)) / 100, 1.0)

    # 3. Fractions  (e.g. "3.5/5")
    m = _RE_FRACTION.match(s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return min(num / den, 1.0) if den > 0 else np.nan

    # 4. Plain numeric string  (e.g. "8" → 0.8)
    try:
        val = float(s)
        if val <= 10:
            return val / 10
    except ValueError:
        pass

    return np.nan


# ===========================================================================
# DATAFRAME INITIALISATION
# ===========================================================================

def init_reviews_df(movies_df: pd.DataFrame, critic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge critic reviews with movie metadata and produce a clean
    reviews DataFrame with normalised scores on a 0-10 scale.

    Steps
    -----
    1. Merge on ``rotten_tomatoes_link`` and keep only relevant columns.
    2. Drop rows missing critic name or movie link.
    3. Parse and normalise review scores; drop rows with unparseable scores.
    4. Rescale to 0-10 and round to one decimal place.

    Returns
    -------
    pd.DataFrame
        Columns: ``critic_name``, ``rotten_tomatoes_link``, ``final_score``
    """
    df = pd.merge(critic_df, movies_df, on="rotten_tomatoes_link")[
        ["critic_name", "rotten_tomatoes_link", "review_type", "review_score"]
    ]

    # Remove rows that lack a critic name or a movie identifier
    df = df.dropna(subset=["critic_name", "rotten_tomatoes_link"])

    # Normalise scores to [0, 1]; drop rows where parsing fails
    df["clean_score"] = df["review_score"].map(parse_score)
    df = df.dropna(subset=["clean_score"])

    # Rescale to 0-10
    df["final_score"] = (df["clean_score"] * 10).round(1)

    return df[["critic_name", "rotten_tomatoes_link", "final_score"]]


# ===========================================================================
# FEATURE ENGINEERING — CATEGORICAL / SPARSE
# ===========================================================================

def tokenize_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Split a comma-separated string column into lists of clean tokens.

    Empty strings and whitespace-only tokens are removed.

    Parameters
    ----------
    df          : DataFrame containing the column to tokenise.
    column_name : Name of the column to process.

    Returns
    -------
    pd.Series of lists
    """
    return df[column_name].fillna("").str.split(",").apply(
        lambda tokens: [t.strip() for t in tokens if t.strip()]
    )


def get_frequent_tokens(series_of_lists: pd.Series, min_freq: int) -> set:
    """
    Return the set of tokens that appear at least ``min_freq`` times
    across all lists in ``series_of_lists``.
    """
    counts = series_of_lists.explode().value_counts()
    return set(counts[counts >= min_freq].index)


def apply_whitelist(series_of_lists: pd.Series, whitelist: set) -> pd.Series:
    """Remove tokens not present in ``whitelist`` from every list."""
    return series_of_lists.apply(lambda lst: [t for t in lst if t in whitelist])


def vectorize_list_column(series_of_lists: pd.Series, index: pd.Index) -> pd.DataFrame:
    """
    Transform a column of token lists into a sparse binary DataFrame
    via ``MultiLabelBinarizer``.

    Parameters
    ----------
    series_of_lists : pd.Series of lists (e.g. genres per movie).
    index           : Index to assign to the resulting DataFrame.

    Returns
    -------
    pd.DataFrame (sparse)  — shape (n_samples, n_unique_tokens)
    """
    mlb = MultiLabelBinarizer(sparse_output=True)
    matrix = mlb.fit_transform(series_of_lists)

    return pd.DataFrame.sparse.from_spmatrix(
        matrix,
        columns=mlb.classes_,
        index=index,
    )


# ===========================================================================
# FEATURE ENGINEERING — DENSE EMBEDDINGS
# ===========================================================================

def build_embedding_matrix(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Encode a text column into a dense embedding matrix using a
    sentence-transformer model.

    Each row of the returned DataFrame corresponds to a movie; each column
    represents one latent dimension of the embedding space (384 dims for
    ``all-MiniLM-L6-v2``).

    Parameters
    ----------
    df          : DataFrame with a ``rotten_tomatoes_link`` column and the
                  target text column.
    text_column : Name of the column containing text to encode (e.g.
                  ``movie_info`` or ``critics_consensus``).

    Returns
    -------
    pd.DataFrame — index: ``rotten_tomatoes_link``, columns: embedding dims
    """
    sentences = df[text_column].fillna("No description available").tolist()

    # encode() shows a progress bar because this can be slow for large corpora
    embeddings = _EMBEDDING_MODEL.encode(sentences, show_progress_bar=True)

    return pd.DataFrame(embeddings, index=df["rotten_tomatoes_link"])


# ===========================================================================
# PERSISTENCE
# ===========================================================================

def save_processed_assets(
        movies_df: pd.DataFrame,
        critic_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
        content_matrix: pd.DataFrame,
        folder_path: str = PROCESSED_DIR,
) -> None:
    """
    Persist all processed assets to disk.

    - Dense DataFrames are saved as Parquet (efficient columnar format).
    - ``content_matrix`` may be a *sparse* DataFrame, which Parquet cannot
      serialise; it is saved as a Pickle file instead.

    Parameters
    ----------
    movies_df      : Cleaned movie metadata.
    critic_df      : Cleaned critic information.
    reviews_df     : Normalised review scores.
    content_matrix : Sparse binary feature matrix (genres, directors, …).
    folder_path    : Destination directory (created if it does not exist).
    """
    os.makedirs(folder_path, exist_ok=True)
    print(f"Saving assets to '{folder_path}' …")

    movies_df.to_parquet(f"{folder_path}/movies_clean.parquet",  index=False)
    critic_df.to_parquet(f"{folder_path}/critic_clean.parquet",  index=False)
    reviews_df.to_parquet(f"{folder_path}/reviews_clean.parquet", index=False)

    # Pickle preserves the exact in-memory object (including sparsity metadata)
    content_matrix.to_pickle(f"{folder_path}/content_matrix.pkl")

    print("All assets saved. (content_matrix stored as .pkl)")


def load_processed_assets(
        folder_path: str = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Load all processed assets from disk.

    Returns
    -------
    (movies_df, critic_df, reviews_df, content_matrix)
        or ``None`` if loading fails.
    """
    print(f"Loading assets from '{folder_path}' …")
    try:
        movies  = pd.read_parquet(f"{folder_path}/movies_clean.parquet")
        critic  = pd.read_parquet(f"{folder_path}/critic_clean.parquet")
        reviews = pd.read_parquet(f"{folder_path}/reviews_clean.parquet")
        matrix  = pd.read_pickle(f"{folder_path}/content_matrix.pkl")

        print("All assets loaded. System ready.")
        return movies, critic, reviews, matrix

    except Exception as exc:
        print(f"ERROR while loading assets: {exc}")
        return None