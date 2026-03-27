import os
import numpy as np
import re
from scipy.sparse import csr_matrix
import kagglehub
import pandas as pd
import joblib
import pyarrow
import fastparquet
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from src.book_one_utils import *
from src.book_three_utils import *

def build_feature_matrix(
        movies_df: pd.DataFrame,
        threshold_dir: int = 3,
        threshold_prod: int = 2,         # Nuova soglia per le case di produzione
        weight_embeddings: float = 2.0,
        weight_genres: float = 0.58,
        weight_directors: float = 0.17,
        weight_production: float = 0.08  # Il peso della "Vibe" A24/Ghibli/Marvel
):
    # 1. Preprocessing & Tokenization
    genres_list = tokenize_column(movies_df, 'genres')
    dirs_list = tokenize_column(movies_df, 'directors')

    # NUOVO: Tokenizziamo le case di produzione
    prod_list = tokenize_column(movies_df, 'production_company')

    # 2. Filtering
    valid_dirs = get_frequent_tokens(dirs_list, threshold_dir)
    dirs_filtered = apply_whitelist(dirs_list, valid_dirs)

    # NUOVO: Filtriamo le case di produzione troppo piccole/sconosciute
    valid_prods = get_frequent_tokens(prod_list, threshold_prod)
    prods_filtered = apply_whitelist(prod_list, valid_prods)

    # 3. Vectorization
    movie_links = movies_df['rotten_tomatoes_link']
    df_genres = vectorize_list_column(genres_list, movie_links)
    df_dirs = vectorize_list_column(dirs_filtered, movie_links)

    # NUOVO: Vettorializziamo le produzioni
    df_prods = vectorize_list_column(prods_filtered, movie_links)

    # 4. Semantic Embedding
    df_embeddings = build_embedding_matrix(movies_df, 'movie_info')
    df_embeddings.index = movie_links

    # 5. Weighting
    weighted_embeddings = df_embeddings * weight_embeddings
    weighted_genres = df_genres * weight_genres
    weighted_dirs = df_dirs * weight_directors

    # NUOVO: Scaliamo il peso delle produzioni
    weighted_prods = df_prods * weight_production

    # Unione delle QUATTRO componenti
    matrix = pd.concat([weighted_genres, weighted_dirs, weighted_prods, weighted_embeddings], axis=1)

    # --- SANIFICAZIONE ---
    if '' in matrix.columns:
        matrix = matrix.drop(columns=[''])

    if not matrix.columns.is_unique:
        matrix = matrix.loc[:, ~matrix.columns.duplicated()]

    return matrix

def book_one():
    download_dataset()
    critic_df, movies_df = create_dataframes()
    dataframe_rw = init_reviews_df(movies_df,critic_df)
    content_matrix = build_feature_matrix(movies_df, threshold_dir=3, weight_genres=2.0)
    save_processed_assets(
        movies_df=movies_df,
        critic_df=critic_df,
        reviews_df=dataframe_rw,
        content_matrix=content_matrix
    )


def get_api_recs(target_movie, metadata, matrix, top_k=10):
    """
    Versione per API: Restituisce solo i dati in formato JSON serializzabile.
    """
    try:
        # 1. Inferenza pura
        results = get_recommendations(target_movie, metadata, matrix, top_k=top_k)

        # 2. Selezione campi necessari per il frontend
        # (Qui non ci serve l'audit dei critici, servono i dati per l'utente)
        cols = ['movie_title', 'production_company', 'similarity_score', 'final_ranking_score']

        # 3. Conversione in JSON serializzabile (lista di dict)
        return results[cols].to_dict(orient="records")

    except Exception:
        return {"error": f"Film '{target_movie}' non trovato o errore interno."}

#def book_two():


def main():
    book_one()
