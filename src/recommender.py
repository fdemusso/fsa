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


def build_feature_matrix(movies_df, threshold_dir=3, weight_genres=2.0, weight_embeddings=1.5):
    """
    Costruisce la matrice finale integrando metadati strutturati e semantica NLP.
    Risolve i conflitti di nomi duplicati e colonne vuote per il salvataggio Parquet.
    """

    # 1. Preprocessing & Tokenization (Assicurati che tokenize_column filtri già gli empty string)
    genres_list = tokenize_column(movies_df, 'genres')
    dirs_list = tokenize_column(movies_df, 'directors')

    # 2. Filtering Registi (Soglia statistica)
    valid_dirs = get_frequent_tokens(dirs_list, threshold_dir)
    dirs_filtered = apply_whitelist(dirs_list, valid_dirs)

    # 3. Vectorization (Generi e Registi - Matrici Sparse)
    # L'indice DEVE essere lo stesso per tutti per evitare disallineamenti e NaN
    movie_links = movies_df['rotten_tomatoes_link']
    df_genres = vectorize_list_column(genres_list, movie_links)
    df_dirs = vectorize_list_column(dirs_filtered, movie_links)

    # 4. Semantic Embedding (Descrizioni - Matrice Densa)
    df_embeddings = build_embedding_matrix(movies_df, 'movie_info')
    # Forziamo l'indice per sicurezza millimetrica
    df_embeddings.index = movie_links

    # 5. Weighting & Concatenation
    weighted_genres = df_genres * weight_genres
    weighted_embeddings = df_embeddings * weight_embeddings

    # Unione delle tre componenti
    matrix = pd.concat([weighted_genres, df_dirs, weighted_embeddings], axis=1)

    # --- SANIFICAZIONE POST-CONCAT (Indispensabile per il salvataggio) ---

    # Rimuoviamo la colonna vuata '' se presente (causa del tuo ValueError)
    if '' in matrix.columns:
        matrix = matrix.drop(columns=[''])

    # Gestione Duplicati: se un regista ha lo stesso nome di un genere,
    # teniamo solo la prima colonna per evitare conflitti nel formato Parquet.
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

#def book_two():


def main():
    book_one()
