"""Decision AI — Feature Engineering Pipeline
=============================================

Este módulo converte as tabelas processadas (`dim_applicant`, `dim_job`,
`fact_prospect`) em matrizes numéricas adequadas para aprendizado de máquina. A
estratégia combina **SBERT** para representar currículos, **TF‑IDF** para a
descrição das vagas e features categóricas/numéricas complementares. O resultado
é um `scikit-learn Pipeline` persistido juntamente com as matrizes ``X`` e
``y`` para uso nos módulos de treinamento.

Usage (CLI):
    $ python -m decision_ai.features.engineer
    # opções avançadas:
    $ python -m decision_ai.features.engineer --tfidf-dim 3000 --svd-dim 512
"""

"""
Feature Engineering Pipeline for Decision AI.

This module transforms processed tables (`dim_applicant`, `dim_job`, `fact_prospect`)
into numerical feature matrices suitable for machine learning. It integrates
SBERT embeddings for CVs, TF-IDF for job descriptions, and one-hot plus
numerical passthrough features, composed into a scikit-learn `Pipeline`.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Install `sentence-transformers` to generate embeddings.") from exc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
FEAT_DIR = DATA_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42


# ───────────────────────────────── helpers ────────────────────────────────
def _load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed Parquet tables.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Tuple containing (dim_applicant, dim_job, fact_prospect) DataFrames.
    """
    dim_app = pd.read_parquet(DATA_DIR / "dim_applicant.parquet")
    dim_job = pd.read_parquet(DATA_DIR / "dim_job.parquet")
    fact = pd.read_parquet(DATA_DIR / "fact_prospect.parquet")
    return dim_app, dim_job, fact


def _status_to_label(status: str) -> int:
    """
    Convert prospect status to binary label.

    Parameters
    ----------
    status : str
        Status string from fact_prospect, e.g., "Contratado".

    Returns
    -------
    int
        1 if status contains "Contratado", otherwise 0.
    """
    return 1 if "Contratado" in status else 0


# ───────────────────────── SBERT wrapper (sklearn) ────────────────────────
class SBERTEncoder:
    """
    SBERT encoder wrapper for integration in scikit-learn pipelines.

    Attributes
    ----------
    model_name : str
        Name of the SBERT model to load.
    _model : SentenceTransformer | None
        Lazy-loaded SentenceTransformer instance.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def fit(self, X: pd.Series, y=None):  # noqa: N802
        """
        No-op fit method to comply with scikit-learn API.

        Parameters
        ----------
        X : pandas.Series
            Input data (ignored).
        y : any, optional
            Target labels (ignored).

        Returns
        -------
        SBERTEncoder
            Self.
        """
        return self  # no‑op

    def transform(self, X: pd.Series):  # noqa: N802
        """
        Compute SBERT embeddings for a series of texts.

        Parameters
        ----------
        X : pandas.Series
            Series of input texts to encode.

        Returns
        -------
        numpy.ndarray
            2D array of normalized SBERT embeddings.
        """
        if self._model is None:
            logger.info("Loading SBERT model “%s”…", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        embeds = []
        with tqdm(total=len(X), desc="embeddings", unit="doc", leave=False) as bar:
            batch_size = 32
            for start in range(0, len(X), batch_size):
                batch = X.iloc[start : start + batch_size].fillna("").tolist()
                emb = self._model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                embeds.append(emb)
                bar.update(len(batch))
        return np.vstack(embeds)

    # compatibility with GridSearch
    def get_params(self, deep: bool = False):
        """
        Get parameters for grid search compatibility.

        Parameters
        ----------
        deep : bool, optional
            Ignored. Included to match scikit-learn signature.

        Returns
        -------
        dict
            Dictionary with key "model_name".
        """
        return {"model_name": self.model_name}

    def set_params(self, **params):
        """
        Set parameters for grid search compatibility.

        Parameters
        ----------
        **params : dict
            Parameters to set on the encoder.

        Returns
        -------
        SBERTEncoder
            Self.
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ────────────────────────── pipeline factory ─────────────────────────────
def build_pipeline(
    tfidf_dim: int = 2000,
    svd_dim: Optional[int] = 256,
    cat_cols: Optional[list[str]] = None,
    num_cols: Optional[list[str]] = None,
) -> Pipeline:
    """
    Create a scikit-learn feature engineering pipeline.

    Parameters
    ----------
    tfidf_dim : int, optional
        Maximum vocabulary size for TF-IDF (default: 2000).
    svd_dim : int or None, optional
        Number of components for TruncatedSVD (None to disable).
    cat_cols : list of str, optional
        List of categorical column names.
    num_cols : list of str, optional
        List of numerical column names.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Configured pipeline combining text, embedding, categorical, and numeric features.
    """
    if cat_cols is None:
        cat_cols = [
            "base__tipo_contratacao",
            "profile__nivel_profissional",
            "profile__nivel_academico",
            "profile__nivel_ingles",
        ]
    if num_cols is None:
        num_cols = ["informacoes_profissionais__remuneracao"]

    # Use tokens em nível de palavra em vez de trigramas de caracteres
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),      # unigram + bigram
        max_features=tfidf_dim,
        min_df=3,                # ignora ruído raro
        stop_words=None,         # pt+en misto; custom stoplist pode ser plugada depois
    )
    sbert = SBERTEncoder()

    coltrans = ColumnTransformer(
        [
            ("tfidf_job", tfidf, "job_text"),
            ("sbert_cv", sbert, "cv_text"),
            ("onehot_cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num_passthrough", "passthrough", num_cols),
        ],
        sparse_threshold=0.3,
    )

    steps = [("coltrans", coltrans)]
    if svd_dim:
        steps.append(("svd", TruncatedSVD(n_components=svd_dim, random_state=SEED)))

    return Pipeline(steps)


# ───────────────────────── FeatureEngineer class ─────────────────────────
class FeatureEngineer:
    """
    Orchestrates building feature matrices from processed DataFrames.

    Attributes
    ----------
    out_dir : pathlib.Path
        Directory to save feature artifacts.
    pipeline : sklearn.pipeline.Pipeline or None
        Fitted pipeline after transformation.
    """
    def __init__(self, out_dir: Path = FEAT_DIR):
        """
        Initialize FeatureEngineer.

        Parameters
        ----------
        out_dir : pathlib.Path, optional
            Directory where feature artifacts will be stored.
        """
        self.out_dir = out_dir
        self.pipeline: Pipeline | None = None

    def build_dataframe(self) -> pd.DataFrame:
        """
        Load and merge processed tables into a single DataFrame with text columns.

        Returns
        -------
        pandas.DataFrame
            Merged DataFrame ready for feature extraction, including 'cv_text',
            'job_text', and binary 'label'.
        """
        logger.info("Carregando tabelas processadas…")
        dim_app, dim_job, fact = _load_tables()
        
        # Harmonize key dtypes before merging (avoid int vs. object)
        for df_, col in (
            (fact, "applicant_id"),
            (fact, "job_id"),
            (dim_app, "applicant_id"),
            (dim_job, "job_id"),
        ):
            if col in df_.columns:
                df_[col] = df_[col].astype("string")

        fact["label"] = fact["status"].apply(_status_to_label)

        df = (
            fact.merge(dim_app, on="applicant_id", how="left")
            .merge(dim_job, on="job_id", how="left")
        )

        # ── column hygiene ───────────────────────────────────────────────
        # Replace spaces and slashes with underscores to stay consistent
        # with the _norm_key() convention applied during ingestion.
        df.columns = (
            df.columns.str.normalize("NFKD")
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
            .str.replace(r"[ /]", "_", regex=True)
            .str.lower()
        )

        # Concatenate and coerce to string to avoid NaNs / floats in text columns
        df["cv_text"] = (
            df.get("cv_pt", pd.Series("", index=df.index)).fillna("").astype(str)
            + " "
            + df.get("cv_en", pd.Series("", index=df.index)).fillna("").astype(str)
        )
        job_fields = [
            "profile__principais_atividades",
            "profile__competencia_tecnicas_e_comportamentais",
        ]
        existing_job_fields = [c for c in job_fields if c in df.columns]

        if not existing_job_fields:
            alt_options = [
                "informacoes_basicas__titulo_vaga",
                "informacoes_basicas__descricao_vaga",
            ]
            alt_field = next((c for c in alt_options if c in df.columns), None)
            if alt_field:
                logger.warning(
                    "Nenhuma das colunas %s presente; usando %s como fallback para job_text",
                    job_fields,
                    alt_field,
                )
                df["job_text"] = df[alt_field].fillna("").astype(str)
            else:
                raise ValueError(
                    "Nenhuma coluna de descricao de vaga encontrada. Esperado pelo menos uma de %s"
                    % job_fields
                )
        else:
            df["job_text"] = (
                df.get(job_fields[0], pd.Series("", index=df.index)).fillna("").astype(str)
                + " "
                + df.get(job_fields[1], pd.Series("", index=df.index)).fillna("").astype(str)
            )

        # Ensure final text columns are pure strings (no floats)
        df["cv_text"] = df["cv_text"].astype(str)
        df["job_text"] = df["job_text"].astype(str)

        if "informacoes_profissionais__remuneracao" in df.columns:
            df["informacoes_profissionais__remuneracao"] = (
                pd.to_numeric(df["informacoes_profissionais__remuneracao"], errors="coerce")
                .fillna(0.0)
            )

        logger.info("DataFrame pronto: %d linhas | positivos=%d", len(df), df["label"].sum())
        return df

    def fit_transform(self, df: pd.DataFrame, tfidf_dim=2000, svd_dim: Optional[int] = 256):
        """
        Fit the feature pipeline and transform the DataFrame into feature matrix.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with merged features and text columns.
        tfidf_dim : int, optional
            Maximum TF-IDF vocabulary size.
        svd_dim : int or None, optional
            Number of SVD components.

        Returns
        -------
        numpy.ndarray
            Feature matrix including TF-IDF, SBERT embeddings, categorical,
            numeric features, and cosine similarity.
        """
        # Determine which categorical and numerical columns actually exist
        cat_default = [
            "base__tipo_contratacao",
            "profile__nivel_profissional",
            "profile__nivel_academico",
            "profile__nivel_ingles",
        ]
        num_default = ["informacoes_profissionais__remuneracao"]

        # Validate that we actually have some text data to vectorize
        for text_col in ["job_text", "cv_text"]:
            if text_col in df.columns and df[text_col].astype(str).str.strip().eq("").all():
                raise ValueError(
                    f"Coluna {text_col} vazia: verifique se os dados foram ingeridos corretamente"
                )

        cat_cols = [c for c in cat_default if c in df.columns]
        num_cols = [c for c in num_default if c in df.columns]

        for missing_cat in set(cat_default) - set(cat_cols):
            logger.warning("Coluna categórica ausente: %s (será ignorada)", missing_cat)
        for missing_num in set(num_default) - set(num_cols):
            logger.warning("Coluna numérica ausente: %s (será ignorada)", missing_num)

        self.pipeline = build_pipeline(
            tfidf_dim,
            svd_dim,
            cat_cols=cat_cols,
            num_cols=num_cols,
        )
        logger.info("Ajustando pipeline de features…")
        X = self.pipeline.fit_transform(df)
        # feature de similaridade de cosseno entre cv_text e job_text
        sbert = SBERTEncoder()
        cv_emb = sbert.transform(df['cv_text'])
        job_emb = sbert.transform(df['job_text'])
        # normaliza embeddings
        cv_norm = cv_emb / np.linalg.norm(cv_emb, axis=1, keepdims=True)
        job_norm = job_emb / np.linalg.norm(job_emb, axis=1, keepdims=True)
        sim = np.sum(cv_norm * job_norm, axis=1).reshape(-1, 1)
        # anexa a feature de similaridade
        X = np.hstack([X, sim])
        return X

    def save(self, X: np.ndarray, y: np.ndarray, df_text: pd.DataFrame | None = None):
        """
        Save feature matrices and artifacts to disk with timestamp.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Target labels array.
        df_text : pandas.DataFrame or None, optional
            Optional DataFrame of text columns to export.
        """
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        joblib.dump(X, self.out_dir / f"X_{ts}.joblib")
        joblib.dump(y, self.out_dir / f"y_{ts}.joblib")
        if df_text is not None:
            df_path = self.out_dir / f"DF_{ts}.parquet"
            df_text.to_parquet(df_path, index=False)
            logger.info("Text DataFrame salvo em %s", df_path)
        joblib.dump(self.pipeline, self.out_dir / f"pipeline_{ts}.joblib")
        logger.info("Artefatos salvos em %s", self.out_dir)

    # -------- convenient entrypoint --------
    @staticmethod
    def run(
        tfidf_dim: int = 2000,
        svd_dim: Optional[int] = 256,
        export_text: bool = False,
    ):
        """
        Execute end-to-end feature engineering and save outputs.

        Parameters
        ----------
        tfidf_dim : int, optional
            Maximum TF-IDF vocabulary size.
        svd_dim : int or None, optional
            Number of SVD components.
        export_text : bool, optional
            Whether to save the text DataFrame for downstream use.
        """
        logger.info("Iniciando engenharia de features…")
        fe = FeatureEngineer()
        df = fe.build_dataframe()
        X = fe.fit_transform(df, tfidf_dim=tfidf_dim, svd_dim=svd_dim)
        text_df = df[["cv_text", "job_text"]] if export_text else None
        fe.save(X, df["label"].values, df_text=text_df)


# ---------------------------------------------------------------------------
# CLI entry-point (argparse)
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parse command-line arguments for feature engineering.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes tfidf_dim, svd_dim, and export_text.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="decision_ai.features.engineer",
        description="Run Decision AI feature‑engineering pipeline.",
    )
    parser.add_argument(
        "--tfidf-dim",
        type=int,
        default=2000,
        help="Maximum TF‑IDF vocabulary size (default: 2000).",
    )
    parser.add_argument(
        "--svd-dim",
        type=int,
        default=256,
        help="Output dimension for TruncatedSVD (use 0 to disable).",
    )
    parser.add_argument(
        "--export-text",
        action="store_true",
        help="Salva DF_<timestamp>.parquet com colunas cv_text e job_text para CatBoost.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    """
    CLI entry point for feature engineering.

    Parses arguments and invokes FeatureEngineer.run().
    """
    args = _parse_args()
    svd = args.svd_dim if args.svd_dim > 0 else None
    FeatureEngineer.run(
        tfidf_dim=args.tfidf_dim,
        svd_dim=svd,
        export_text=args.export_text,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
