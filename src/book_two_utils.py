"""
book_two_utils.py
==================
EDA utilities for Book 2.

Focus:
  - Review score distribution analysis (S_final)
  - Content matrix sparsity / entropy / multicollinearity
  - Dimensionality reduction diagnostics (SVD + t-SNE)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import warnings
import os

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.sparse import issparse, csr_matrix, spmatrix
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.book_one_utils import load_processed_assets, PROCESSED_DIR, save_processed_assets

# ===========================================================================
# CONFIGURATION
# ===========================================================================

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_eda_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed assets and return only objects relevant for Book 2 EDA.

    Returns
    -------
    reviews_df : pd.DataFrame
        Clean reviews with 'final_score'.
    content_matrix : pd.DataFrame
        Hybrid feature matrix used by the recommender pipeline.
    """
    required_files = [
        os.path.join(PROCESSED_DIR, "movies_clean.parquet"),
        os.path.join(PROCESSED_DIR, "critic_clean.parquet"),
        os.path.join(PROCESSED_DIR, "reviews_clean.parquet"),
        os.path.join(PROCESSED_DIR, "content_matrix.pkl"),
    ]
    missing_assets = [path for path in required_files if not os.path.isfile(path)]

    if missing_assets:
        # Bootstrap: rebuild processed data when persistence layer is missing.
        from src.recommender import book_one

        print("Asset processati non trovati. Avvio pipeline Book 1...")
        book_one()

    loaded_assets = load_processed_assets()
    if loaded_assets is None:
        raise RuntimeError("Impossibile caricare gli asset processati da ../data/processed")

    _, _, reviews_df, content_matrix = loaded_assets
    return reviews_df, content_matrix


def get_score_series(reviews_df: pd.DataFrame, score_col: str = "final_score") -> pd.Series:
    """Return a clean numeric score series for S_final analysis."""
    if score_col not in reviews_df.columns:
        raise RuntimeError(f"La colonna '{score_col}' non è presente in reviews_df.")

    s_final = pd.to_numeric(reviews_df[score_col], errors="coerce").dropna()
    if s_final.empty:
        raise RuntimeError("S_final è vuoto dopo la pulizia dei NaN.")

    return s_final


def to_csr_matrix(content_matrix: pd.DataFrame | np.ndarray | spmatrix) -> tuple[csr_matrix, list]:
    """Convert content matrix to CSR and return matrix + feature names."""
    if isinstance(content_matrix, pd.DataFrame):
        if hasattr(content_matrix, "sparse"):
            matrix_values = content_matrix.sparse.to_coo().tocsr()
        else:
            matrix_values = csr_matrix(content_matrix.values)
        feature_names = list(content_matrix.columns)
    elif isinstance(content_matrix, spmatrix):
        matrix_values = content_matrix.tocsr()
        feature_names = [f"f_{i}" for i in range(matrix_values.shape[1])]
    else:
        matrix_values = csr_matrix(content_matrix)
        feature_names = [f"f_{i}" for i in range(matrix_values.shape[1])]

    return matrix_values, feature_names


# ===========================================================================
# 1) PDF, CDF, MOMENTS
# ===========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def compute_statistical_moments(s_final: pd.Series) -> pd.DataFrame:
    """
    Compute statistical moments (mean, variance, skewness, kurtosis)
    and perform normality tests (Shapiro-Wilk, Kolmogorov-Smirnov).

    Returns
    -------
    pd.DataFrame
        Table with exact statistical metrics.
    """
    mu = float(s_final.mean())
    var = float(s_final.var(ddof=1))
    skewness = float(stats.skew(s_final, bias=False))
    kurtosis_excess = float(stats.kurtosis(s_final, fisher=True, bias=False))
    kurtosis_pearson = kurtosis_excess + 3.0

    # Sub-sampling per Shapiro-Wilk (richiesto per N > 5000)
    sample = s_final.sample(n=min(5000, len(s_final)), random_state=42)
    _, shapiro_p = stats.shapiro(sample)

    # Standardizzazione per test K-S
    sample_z = ((sample - sample.mean()) / sample.std(ddof=1)).to_numpy()
    _, ks_p = stats.kstest(sample_z, "norm")

    return pd.DataFrame(
        {
            "metric": [
                "mean",
                "variance",
                "skewness",
                "kurtosis_excess",
                "kurtosis_pearson",
                "shapiro_p_value",
                "ks_p_value",
            ],
            "value": [mu, var, skewness, kurtosis_excess, kurtosis_pearson, shapiro_p, ks_p],
        }
    )

def plot_pdf_distribution(s_final: pd.Series) -> None:
    """
    Visualize the empirical Probability Density Function (Hist + KDE)
    against the theoretical Normal distribution.
    """
    mu = float(s_final.mean())
    std = float(s_final.std(ddof=1))

    x_grid = np.linspace(s_final.min(), s_final.max(), 400)
    normal_pdf = stats.norm.pdf(x_grid, loc=mu, scale=std)

    plt.figure(figsize=(11, 5))
    sns.histplot(s_final, bins=40, stat="density", color="#4c72b0", alpha=0.25, edgecolor=None)
    sns.kdeplot(s_final, color="#dd8452", linewidth=2.3, label="KDE (PDF empirica)")
    plt.plot(x_grid, normal_pdf, color="#55a868", linewidth=2.0, linestyle="--", label="PDF Normale (fit)")

    plt.title("Distribuzione di S_final: Hist + KDE + Normal fit")
    plt.xlabel("S_final")
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cdf_distribution(s_final: pd.Series) -> None:
    """
    Visualize the Empirical Cumulative Distribution Function (ECDF)
    against the theoretical Normal CDF to detect step-wise discretization.
    """
    mu = float(s_final.mean())
    std = float(s_final.std(ddof=1))

    x_sorted = np.sort(s_final.values)
    y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    normal_cdf = stats.norm.cdf(x_sorted, loc=mu, scale=std)

    plt.figure(figsize=(11, 5))
    plt.plot(x_sorted, y_ecdf, color="#c44e52", linewidth=2.3, label="ECDF empirica")
    plt.plot(x_sorted, normal_cdf, color="#8172b2", linewidth=2.0, linestyle="--", label="CDF Normale (fit)")

    plt.title("CDF di S_final")
    plt.xlabel("x")
    plt.ylabel("P(X <= x)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_sufficiency_probability(s_final: pd.Series, sufficiency_threshold: float = 6.0) -> pd.DataFrame:
    """
    Calculate the cumulative probability threshold for the recommendation matrix constraint.

    Returns
    -------
    pd.DataFrame
        Single-row dataframe with the probability metric.
    """
    p_below = float((s_final <= sufficiency_threshold).mean())

    return pd.DataFrame(
        {
            "metric": [f"P(S_final <= {sufficiency_threshold:.1f})"],
            "value": [p_below]
        }
    )
# ===========================================================================
# 2) SPARSITY + SHANNON ENTROPY
# ===========================================================================

def analyze_sparsity_and_entropy(content_matrix: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """
    Compute matrix sparsity and per-feature Shannon entropy.

    Returns
    -------
    pd.DataFrame
        Features sorted by increasing entropy.
    """
    matrix_values, feature_names = to_csr_matrix(content_matrix)

    n_rows, n_cols = matrix_values.shape
    total_elements = n_rows * n_cols
    non_zero = int(matrix_values.nnz)
    sparsity = 1.0 - (non_zero / total_elements)

    print(f"Matrice shape: ({n_rows}, {n_cols})")
    print(f"Elementi non nulli: {non_zero}")
    print(f"Sparsity S: {sparsity:.6f}")

    binary_presence = matrix_values.copy()
    binary_presence.data = np.ones_like(binary_presence.data)

    p1 = np.asarray(binary_presence.mean(axis=0)).ravel()
    p0 = 1.0 - p1

    eps = 1e-12
    entropy = -(p0 * np.log2(p0 + eps) + p1 * np.log2(p1 + eps))

    entropy_df = pd.DataFrame(
        {"feature": feature_names, "p_presence": p1, "entropy": entropy}
    ).sort_values("entropy", ascending=True)

    print("\nFeature a bassa entropia (meno discriminanti):")
    print(entropy_df.head(20))

    plt.figure(figsize=(11, 5))
    sns.histplot(entropy_df["entropy"], bins=40, color="#4c72b0")
    plt.title("Distribuzione entropia di Shannon per feature")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    return entropy_df


# ===========================================================================
# 3) CORRELATION + VIF
# ===========================================================================

def _extract_categorical_binary(content_matrix: pd.DataFrame, max_features: int) -> pd.DataFrame:
    """
    Internal helper to extract, densify, and binarize the categorical features.
    Elimina le dipendenze esterne fantasma e lavora nativamente sui tipi sparsi di Pandas.
    """
    # 1. Isolamento Architetturale: Vogliamo SOLO generi e registi.
    # Sappiamo che gli embeddings SBERT hanno nomi colonna numerici (es. 0, 1... 383)
    # mentre i generi/registi sono stringhe non numeriche (es. 'Action', 'Tarantino').
    cat_cols = [c for c in content_matrix.columns if isinstance(c, str) and not str(c).isdigit()]

    # Fallback se le colonne non sono nomate come previsto
    if len(cat_cols) == 0:
        cat_cols = content_matrix.columns.tolist()

    cat_df = content_matrix[cat_cols].copy()

    # 2. Densificazione Selettiva (necessaria per calcolare Pearson/Spearman/VIF)
    # Pandas gestisce la sparsità a livello di singola colonna.
    for col in cat_df.columns:
        if pd.api.types.is_sparse(cat_df[col]):
            cat_df[col] = cat_df[col].sparse.to_dense()

    # 3. Binarizzazione Logica (Se il peso è > 0, la feature è presente = 1, altrimenti = 0)
    cat_binary = (cat_df > 0).astype(int)

    # 4. Selezione delle feature più frequenti (Taglio della Coda Lunga per le Heatmap)
    feature_freq = cat_binary.mean(axis=0).sort_values(ascending=False)
    selected_cols = feature_freq.head(min(max_features, len(feature_freq))).index.tolist()

    return cat_binary[selected_cols].copy()


def plot_pearson_correlation(content_matrix: pd.DataFrame | np.ndarray, max_features: int = 40) -> None:
    """
    Visualize the Pearson correlation heatmap for the top categorical features.
    Mathematically equivalent to the phi coefficient for binary data.
    """
    corr_df = _extract_categorical_binary(content_matrix, max_features)
    pearson_corr = corr_df.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, cmap="coolwarm", center=0.0, annot=False)
    plt.title("Pearson Correlation Heatmap (Top Categorical Features)")
    plt.tight_layout()
    plt.show()


def plot_spearman_correlation(content_matrix: pd.DataFrame | np.ndarray, max_features: int = 40) -> None:
    """
    Visualize the Spearman rank-order correlation heatmap.
    Useful for detecting monotonic relationships resilient to outliers.
    """
    corr_df = _extract_categorical_binary(content_matrix, max_features)
    spearman_corr = corr_df.corr(method="spearman")

    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_corr, cmap="vlag", center=0.0, annot=False)
    plt.title("Spearman Correlation Heatmap (Top Categorical Features)")
    plt.tight_layout()
    plt.show()


def compute_vif_metrics(content_matrix: pd.DataFrame | np.ndarray, max_features: int = 25) -> pd.DataFrame:
    """
    Compute the Variance Inflation Factor (VIF) to detect multicollinearity.

    Returns
    -------
    pd.DataFrame
        Table of features and their respective VIF scores, sorted descending.
    """
    x_vif = _extract_categorical_binary(content_matrix, max_features)

    # Rimuovi le costanti che farebbero esplodere il calcolo
    non_constant_cols = [c for c in x_vif.columns if x_vif[c].nunique() > 1]
    x_vif = x_vif[non_constant_cols]

    if x_vif.shape[1] < 2:
        print("Insufficient non-constant features to compute VIF.")
        return pd.DataFrame(columns=["feature", "VIF"])

    # Standardizzazione necessaria per un calcolo VIF stabile
    x_vif_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(x_vif)

    vif_df = pd.DataFrame({
        "feature": x_vif.columns,
        "VIF": [variance_inflation_factor(x_vif_scaled, i) for i in range(x_vif_scaled.shape[1])]
    }).sort_values("VIF", ascending=False)

    return vif_df


def plot_vif_distribution(vif_df: pd.DataFrame, threshold: float = 5.0) -> None:
    """
    Visualize the VIF scores using a barplot, highlighting the critical threshold.
    """
    if vif_df.empty:
        print("VIF DataFrame is empty. Cannot plot.")
        return

    plt.figure(figsize=(11, 6))
    sns.barplot(data=vif_df.head(20), x="VIF", y="feature", palette="rocket")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Critical Threshold ({threshold})")
    plt.title("Variance Inflation Factor (Top 20 Features)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 4) DIMENSIONALITY REDUCTION
# ===========================================================================

def fit_svd_and_plot_variance(
        content_matrix: pd.DataFrame | np.ndarray,
        svd_components: int = 100
) -> np.ndarray:
    """
    Compute TruncatedSVD, plot cumulative explained variance, and return the projected matrix.
    TruncatedSVD is strictly required over PCA because our categorical space is highly sparse.
    """
    # Usiamo la tua funzione o logica per estrarre la matrice
    # Assumiamo che se è DataFrame, i valori numerici siano estratti correttamente
    if isinstance(content_matrix, pd.DataFrame):
        matrix_values = content_matrix.sparse.to_dense().to_numpy() if hasattr(content_matrix, "sparse") else content_matrix.to_numpy()
    else:
        matrix_values = content_matrix

    n_features_total = matrix_values.shape[1]
    n_components = min(svd_components, max(2, n_features_total - 1))

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    x_svd = svd.fit_transform(matrix_values)

    cum_explained = np.cumsum(svd.explained_variance_ratio_)
    components_axis = np.arange(1, len(cum_explained) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(components_axis, cum_explained, marker="o", linewidth=1.8, markersize=3, color="#4c72b0")
    plt.axhline(0.80, color="red", linestyle="--", linewidth=1.5, label="80% Threshold")
    plt.title("Cumulative Explained Variance (TruncatedSVD)")
    plt.xlabel("Number of SVD Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return x_svd

def plot_svd_projection(x_svd: np.ndarray) -> None:
    """
    Visualize the global structure using the first two Principal Components.
    """
    plt.figure(figsize=(9, 7))
    plt.scatter(x_svd[:, 0], x_svd[:, 1], s=10, alpha=0.5, color="#55a868")
    plt.title("Global Geometry: Projection on Top 2 SVD Components")
    plt.xlabel("Component 1 (Highest Variance)")
    plt.ylabel("Component 2 (Second Highest Variance)")
    plt.tight_layout()
    plt.show()

def fit_tsne_and_plot_manifold(x_svd: np.ndarray, tsne_max_samples: int = 3000) -> np.ndarray:
    """
    Compute t-SNE on the top SVD components to discover non-linear local manifolds.
    """
    rng = np.random.default_rng(42)

    # Per stabilità matematica, t-SNE richiede input pre-ridotti (solitamente max 50 dimensioni)
    if x_svd.shape[0] > tsne_max_samples:
        sample_idx = rng.choice(x_svd.shape[0], size=tsne_max_samples, replace=False)
        x_tsne_input = x_svd[sample_idx, : min(50, x_svd.shape[1])]
    else:
        x_tsne_input = x_svd[:, : min(50, x_svd.shape[1])]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
        max_iter=1200,
    )

    x_tsne = tsne.fit_transform(x_tsne_input)

    plt.figure(figsize=(9, 7))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=10, alpha=0.6, color="#dd8452")
    plt.title("Non-linear Semantic Manifold (t-SNE 2D Projection)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.show()

    return x_tsne


# ===========================================================================
# PIPELINE ORCHESTRATION
# ===========================================================================


def generate_projection_coordinates() -> None:
    """
    Procedura automatizzata: Carica gli asset, calcola le coordinate x,y
    tramite TruncatedSVD e sovrascrive i file su disco.
    """
    # 1. CARICAMENTO ASSET (Usando la tua funzione del Book 01)
    assets = load_processed_assets()
    if assets is None:
        print("Impossibile procedere: Asset non trovati.")
        return

    movies_df, critic_df, reviews_df, content_matrix = assets

    # 2. ESTRAZIONE MATRICE
    # Gestiamo la sparsità: se è un DataFrame Pandas con tipi sparsi,
    # dobbiamo densificare o estrarre i valori per SVD.
    if hasattr(content_matrix, "sparse"):
        x_input = content_matrix.sparse.to_dense().values
    elif hasattr(content_matrix, "values"):
        x_input = content_matrix.values
    else:
        x_input = content_matrix

    # 3. RIDUZIONE DIMENSIONALE (SVD)
    # Calcoliamo le coordinate x, y nello spazio latente
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(x_input)

    # 4. INIEZIONE COORDINATE
    movies_df['x'] = coords[:, 0]
    movies_df['y'] = coords[:, 1]

    # 5. PERSISTENZA (Sovrascrittura asset aggiornati)
    save_processed_assets(
        movies_df=movies_df,
        critic_df=critic_df,
        reviews_df=reviews_df,
        content_matrix=content_matrix
    )


