from __future__ import annotations

import joblib
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


def _latest(pattern: str) -> Path:
    files = sorted(MODEL_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern}")
    return files[-1]


def load_latest_artifacts() -> tuple:
    model = joblib.load(_latest("model_*.joblib"))
    pipeline = joblib.load(_latest("pipeline_*.joblib"))
    return model, pipeline


def predict(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    model, pipeline = load_latest_artifacts()
    X = pipeline.transform(df)
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"proba": proba, "pred": (proba >= threshold).astype(int)})
