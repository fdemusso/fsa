import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from IPython.display import display

def compute_similarity_scores(target_idx: int, content_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Calcola la Cosine Similarity tra un singolo vettore target e l'intero spazio vettoriale."""
    matrix_array = content_matrix.values if isinstance(content_matrix, pd.DataFrame) else content_matrix
    target_vector = matrix_array[target_idx].reshape(1, -1)
    similarities = cosine_similarity(target_vector, matrix_array).flatten()
    return similarities

def get_recommendations(
        movie_title: str,
        metadata: pd.DataFrame,
        content_matrix: pd.DataFrame | np.ndarray,
        top_k: int = 10,
        quality_weight = 0.2324,      # Abbassalo se vuoi salvare film come Stargate
        audience_blend = 0.38,# Bilanciamento Critica/Pubblico
        popularity_penalty_min = 0.02782, # Tassa fissa per i blockbuster
        popularity_penalty_max = 0.515 # Mannaia per i blockbuster fuori tema
) -> pd.DataFrame:

    # 1. Ricerca dell'Indice
    idx_list = metadata[metadata['movie_title'].str.lower() == movie_title.lower()].index.tolist()
    if not idx_list:
        raise ValueError(f"Errore: Il film '{movie_title}' non esiste.")
    target_idx = idx_list[0]

    # 2. Distanza Spaziale (Trama + Regista + Genere + Produzione)
    sim_scores = compute_similarity_scores(target_idx, content_matrix)

    # --- 3. IL MOTORE DI QUALITA' IBRIDO (Critica + Pubblico) ---

    # Normalizziamo la Critica (0-1)
    global_critic_mean = metadata['s_final_mean'].mean()
    critic_filled = metadata['s_final_mean'].fillna(global_critic_mean)
    norm_critic = critic_filled / critic_filled.max()

    # Normalizziamo il Pubblico (0-1) Assumiamo sia in base 100, altrimenti usiamo il max
    global_aud_mean = metadata['audience_rating'].mean()
    aud_filled = metadata['audience_rating'].fillna(global_aud_mean)
    # Se i voti sono percentuali, aud_filled.max() sarà vicino a 100
    norm_audience = aud_filled / aud_filled.max()

    # Creiamo il super-voto bilanciato
    combined_quality = (norm_critic * (1.0 - audience_blend)) + (norm_audience * audience_blend)

    # 4. Aggregazione Base
    sim_df = pd.DataFrame({
        'movie_id': metadata.index,
        'movie_title': metadata['movie_title'],
        'production_company': metadata['production_company'], # Aggiunto per controllo visivo
        'similarity_score': sim_scores,
        'norm_critic': norm_critic,
        'norm_audience': norm_audience,
        'combined_quality': combined_quality,
        'audience_count': metadata['audience_count']
    })

    sim_df = sim_df[sim_df['movie_id'] != target_idx]

    # 5. RANKING BASE (Similarità + Super-Qualità)
    sim_df['base_ranking_score'] = (
            (sim_df['similarity_score'] * (1.0 - quality_weight)) +
            (sim_df['combined_quality'] * quality_weight)
    )

    # 6. IL FRENO ANTI-BLOCKBUSTER A RANGE DINAMICO
    aud_count = sim_df['audience_count'].fillna(0)
    log_aud_count = np.log10(aud_count + 1)
    max_log = np.log10(metadata['audience_count'].max() + 1)
    norm_popularity = log_aud_count / max_log if max_log > 0 else 0
    sim_df['norm_popularity'] = norm_popularity

    dynamic_penalty_weight = popularity_penalty_min + (popularity_penalty_max - popularity_penalty_min) * (1.0 - sim_df['similarity_score'])

    # Decurtazione finale
    sim_df['final_ranking_score'] = sim_df['base_ranking_score'] * (1.0 - (sim_df['norm_popularity'] * dynamic_penalty_weight))

    # --- STEP 8: LA MANNAIA STATISTICA (NON LINEARE) ---
    # Recuperiamo i conteggi (gestendo i NaN con 0 per evitare errori matematici)
    c_count = metadata.loc[sim_df.index, 'tomatometer_count'].fillna(0)
    a_count = metadata.loc[sim_df.index, 'audience_count'].fillna(0)

    # 1. Funzione di Confidenza Logaritmica
    # Questi sono i tuoi "punti di saturazione"
    critics_threshold = 50
    audience_threshold = 500

    # Calcoliamo quanto "pesa" il film rispetto alla validità minima
    # Il clip(0, 1) è fondamentale: se un film ha 1000 critici, il valore non deve esplodere a 3.0, deve fermarsi a 1.0
    conf_critics = (np.log10(c_count + 1) / np.log10(critics_threshold + 1)).clip(0, 1)
    conf_audience = (np.log10(a_count + 1) / np.log10(audience_threshold + 1)).clip(0, 1)

    # 2. Logica "OR" (Il massimo della confidenza disponibile)
    # Se il film è un capolavoro di nicchia (tanti critici) o un successo pop (tanto pubblico), passa.
    final_confidence = np.maximum(conf_critics, conf_audience)

    # 3. Applicazione del Fattore di Confidenza
    sim_df['final_ranking_score'] = sim_df['final_ranking_score'] * final_confidence

    # 4. IL COLPO DI GRAZIA (Hard Cut per i Fantasmi)
    # Se sei sotto il minimo sindacale in ENTRAMBI i mondi, il tuo punteggio diventa ZERO.
    ghost_mask = (c_count < 50) & (a_count < 100)
    sim_df.loc[ghost_mask, 'final_ranking_score'] = 0

    # --- STEP 9: ORDINAMENTO DEFINITIVO (SOLO ORA!) ---
    # Adesso che i punteggi sono stati "puliti", possiamo decidere chi merita il podio
    sim_df = sim_df.sort_values(by='final_ranking_score', ascending=False)

    return sim_df.head(top_k)



def run_interactive_engine(metadata, matrix, top_k=20):
    target_movie = input("\n🔍 Inserisci il titolo del film: ").strip()

    if not target_movie:
        return

    try:
        # 1. Calcolo Raccomandazioni
        results = get_recommendations(target_movie, metadata, matrix, top_k=top_k)

        # 2. Audit Colonne
        target_col = 'tomatometer_count' if 'tomatometer_count' in metadata.columns else 'review_count'
        results['critics_count'] = metadata.loc[results.index, target_col].fillna(0)

        cols = ['movie_title', 'production_company', 'critics_count', 'audience_count',
                'similarity_score', 'norm_critic', 'norm_audience', 'final_ranking_score']

        # 3. Display Tabella (Quello che ti piaceva)
        print(f"\n🎯 RISULTATI PER: {target_movie.upper()}")
        display(results[cols])

        # 4. VISUALIZZAZIONE GEOMETRICA (Se le coordinate esistono)
        if 'x' in metadata.columns and 'y' in metadata.columns:
            print("\n🗺️ Generazione Mappa Semantica...")

            # Prepariamo i dati per il plot
            plot_df = metadata.copy()
            plot_df['type'] = 'Altri Film'

            # Marcatura dei risultati e del target
            rec_titles = results['movie_title'].tolist()
            plot_df.loc[plot_df['movie_title'].isin(rec_titles), 'type'] = 'RACCOMANDATI'
            plot_df.loc[plot_df['movie_title'].str.lower() == target_movie.lower(), 'type'] = 'TARGET'

            # Creazione del grafico
            fig = px.scatter(
                plot_df, x='x', y='y',
                color='type',
                hover_name='movie_title',
                color_discrete_map={'Altri Film': 'lightgrey', 'RACCOMANDATI': '#00CC96', 'TARGET': '#EF553B'},
                title=f"Posizionamento Spaziale: {target_movie}",
                template="plotly_dark"
            )
            fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
            display(fig)
        else:
            print("\n⚠️ Coordinate x, y non trovate nel dataset. Salta la visualizzazione grafica.")

    except Exception as e:
        print(f"⚠️ Errore: {e}")

import os
import joblib
import pandas as pd

def export_recommender_model(metadata, matrix, target_path):
    """
    Sincronizza e serializza i componenti core del modello per la produzione.
    """
    # 1. Creazione della directory (se non esiste)
    os.makedirs(target_path, exist_ok=True)
    print(f"📁 Preparazione directory: {target_path}")

    # 2. Esportazione Metadati (con coordinate x, y e metriche di qualità)
    # Usiamo Parquet per mantenere l'efficienza e la precisione dei float
    metadata_file = os.path.join(target_path, "recommender_metadata.parquet")
    metadata.to_parquet(metadata_file, index=True)

    # 3. Esportazione Matrice di Contenuto
    # Joblib è ottimizzato per oggetti Python con grandi array di dati
    matrix_file = os.path.join(target_path, "content_matrix.joblib")
    joblib.dump(matrix, matrix_file)
    print(f"Completato")


