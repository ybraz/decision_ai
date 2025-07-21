"""
Module `decision_ai.models.predict`.

Provides functionality to load the latest model and pipeline artifacts
and generate predictions with probabilities and binary labels for input dataframes.
"""
from __future__ import annotations

import joblib
from pathlib import Path
from typing import List

import pandas as pd

import logging

import numpy as np
from decision_ai.features.engineer import SBERTEncoder

logger = logging.getLogger("decision_ai.models.predict")
logger.setLevel(logging.INFO)
# configura handler para exibir logs INFO
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


def _latest(pattern: str) -> Path:
    """
    Retrieve the most recent file matching a glob pattern in the model directory.

    Parameters
    ----------
    pattern : str
        Glob pattern to match artifact filenames.

    Returns
    -------
    pathlib.Path
        Path to the latest matching file.

    Raises
    ------
    FileNotFoundError
        If no files match the given pattern.
    """
    files = sorted(MODEL_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern}")
    return files[-1]


def load_latest_artifacts() -> tuple:
    """
    Load the most recent trained model and preprocessing pipeline.

    Returns
    -------
    Tuple[object, sklearn.pipeline.Pipeline]
        A tuple containing the trained model and the fitted preprocessing pipeline.
    """
    model = joblib.load(_latest("model_*.joblib"))
    pipeline = joblib.load(_latest("pipeline_*.joblib"))
    return model, pipeline


def predict(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Generate predictions for input data.

    This function transforms the input DataFrame using the preprocessing pipeline,
    computes additional cosine similarity features using SBERT embeddings,
    and applies the trained model to obtain probability scores and binary predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'cv_text' and 'job_text' columns and any
        other features required by the pipeline.
    threshold : float, optional
        Probability threshold for classifying positive predictions (default is 0.5).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'proba' for predicted probabilities of the positive class
        and 'pred' for binary predictions.

    Raises
    ------
    ValueError
        If transformation or prediction fails due to invalid input or threshold.
    """
    model, pipeline = load_latest_artifacts()
    # logando diagnóstico do input
    logger.info(f"Predict recebeu DataFrame com colunas: {df.columns.tolist()}")
    X = pipeline.transform(df)
    # logando forma dos features gerados
    logger.info(f"Features transformadas shape: {X.shape}")
    # calcula feature de similaridade entre cv_text e job_text
    cv_emb = SBERTEncoder().transform(df['cv_text'])
    job_emb = SBERTEncoder().transform(df['job_text'])
    # normaliza embeddings
    cv_norm = cv_emb / np.linalg.norm(cv_emb, axis=1, keepdims=True)
    job_norm = job_emb / np.linalg.norm(job_emb, axis=1, keepdims=True)
    # similaridade de cosseno
    sim = np.sum(cv_norm * job_norm, axis=1).reshape(-1, 1)
    # anexa a feature de similaridade
    X = np.hstack([X, sim])
    logger.info(f"Shape após similaridade: {X.shape}")
    # logando classes do modelo
    logger.info(f"Model.classes_: {model.classes_}")
    # obtendo probabilidades brutas (ambas as classes)
    proba_raw = model.predict_proba(X)
    logger.info(f"Predict_proba raw output: {proba_raw.tolist()}")
    # selecionando probabilidade da classe positiva (índice 1)
    proba = proba_raw[:, 1]
    # estatísticas das probabilidades previstas
    logger.info(f"Probabilidades -> min: {proba.min():.4f}, mean: {proba.mean():.4f}, max: {proba.max():.4f}")
    return pd.DataFrame({"proba": proba, "pred": (proba >= threshold).astype(int)})
