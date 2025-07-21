"""
Module `decision_ai.models.evaluate`.

Provides functionality to load model and feature artifacts and evaluate
performance by computing metrics, generating plots, and exporting reports.
"""
from __future__ import annotations

import joblib
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "processed" / "features"
MODEL_DIR = ROOT / "models"


def _latest(base: Path, pattern: str) -> Path:
    """
    Retrieve the most recent file matching a glob pattern in the given directory.

    Parameters
    ----------
    base : pathlib.Path
        Directory in which to search for files.
    pattern : str
        Glob pattern to match filenames.

    Returns
    -------
    pathlib.Path
        Path to the latest matching file.

    Raises
    ------
    FileNotFoundError
        If no files match the given pattern.
    """
    files = sorted(base.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern}")
    return files[-1]


def load_artifacts() -> tuple:
    """
    Load the most recent trained model, preprocessing pipeline, feature matrix, and labels.

    Returns
    -------
    Tuple[object, sklearn.pipeline.Pipeline, numpy.ndarray, numpy.ndarray]
        Tuple containing the loaded model, pipeline, feature matrix X, and labels y.
    """
    model = joblib.load(_latest(MODEL_DIR, "model_*.joblib"))
    pipeline = joblib.load(_latest(MODEL_DIR, "pipeline_*.joblib"))
    X = joblib.load(_latest(FEAT_DIR, "X_*.joblib"))
    y = joblib.load(_latest(FEAT_DIR, "y_*.joblib"))
    return model, pipeline, X, y


def evaluate(threshold: float = 0.5, export: Optional[Path] = None) -> None:
    """
    Evaluate model performance and optionally export metrics and visualization plots.

    Parameters
    ----------
    threshold : float, optional
        Probability threshold for binary classification (default is 0.5).
    export : pathlib.Path or None, optional
        Directory to save evaluation reports and plots; if None, results are not exported.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If threshold is outside the range [0, 1] or evaluation fails.
    """
    model, pipeline, X, y = load_artifacts()
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    roc = roc_auc_score(y, probs)
    print(classification_report(y, preds))
    print(f"ROC-AUC: {roc:.3f}")

    if export:
        export.mkdir(parents=True, exist_ok=True)
        # ROC curve
        RocCurveDisplay.from_predictions(y, probs).figure_.savefig(export / "roc_curve.png")
        # PR curve
        PrecisionRecallDisplay.from_predictions(y, probs).figure_.savefig(export / "pr_curve.png")
        # report
        (export / "classification_report.txt").write_text(classification_report(y, preds))
        print(f"Reports saved to {export}")


# CLI entry point: parse arguments and invoke model evaluation.
if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Evaluate trained model")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--export", type=Path)
    args = p.parse_args()
    evaluate(threshold=args.threshold, export=args.export)
