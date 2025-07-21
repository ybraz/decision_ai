from __future__ import annotations

import joblib
from datetime import datetime
from pathlib import Path
from typing import Tuple

import optuna
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "processed" / "features"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _latest(pattern: str) -> Path:
    files = sorted(FEAT_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No artifact matching {pattern}")
    return files[-1]


def load_features() -> Tuple[np.ndarray, np.ndarray, object]:
    X = joblib.load(_latest("X_*.joblib"))
    y = joblib.load(_latest("y_*.joblib"))
    pipeline = joblib.load(_latest("pipeline_*.joblib"))
    return X, y, pipeline


def _objective(trial: optuna.Trial, X_train, X_valid, y_train, y_valid) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = LGBMClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=1,
        **params,
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, preds)


def train(n_trials: int = 30) -> Path:
    """Train LightGBM model and save artefact."""
    X, y, pipeline = load_features()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: _objective(t, X_tr, X_val, y_tr, y_val), n_trials=n_trials)

    best_params = study.best_params
    model = LGBMClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=1,
        **best_params,
    )
    model.fit(X, y)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = MODEL_DIR / f"model_{ts}.joblib"
    pipe_path = MODEL_DIR / f"pipeline_{ts}.joblib"
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipe_path)
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Train LGBM model")
    p.add_argument("--trials", type=int, default=30)
    args = p.parse_args()
    train(n_trials=args.trials)
