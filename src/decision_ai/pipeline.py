"""Decision AI — Orquestração de Pipeline (Prefect + Typer CLI)

Este módulo oferece uma *orquestração de ponta a ponta* que encadeia as
etapas:

1. **Ingestão** dos JSONs → Parquet (`ingest.Ingestor`)
2. **Feature Engineering** (`features.engineer.FeatureEngineer`)
3. **Treinamento / HPO** (`models.train.train`)

A orquestração é implementada de duas formas:

* **Prefect Flow** (recomendado em produção)
* **Execução sequencial** simples para rodar localmente ou em CI

Uso (Prefect):
    $ poetry run python -m decision_ai.pipeline run-prefect

Uso (sequencial rápido):
    $ poetry run python -m decision_ai.pipeline run-local
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

try:
    from prefect import flow, task  # real Prefect
    _PREFECT_AVAILABLE = True
except ModuleNotFoundError:  # Prefect not installed → create no‑op stubs
    _PREFECT_AVAILABLE = False

    def flow(*_a, **_kw):  # type: ignore
        def decorator(fn):
            return fn
        return decorator

    def task(*_a, **_kw):  # type: ignore
        def decorator(fn):
            return fn
        return decorator

# Importação tardia para evitar custo de import quando não usado
def _lazy_imports():
    global Ingestor, FeatureEngineer, train  # noqa: PLW0603
    from decision_ai.data.ingest import Ingestor
    from decision_ai.features.engineer import FeatureEngineer
    from decision_ai.models.train import train


logger = logging.getLogger("decision_ai.pipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Prefect Tasks / Flow
# ---------------------------------------------------------------------------


@task(name="Ingest JSONs", retries=1, retry_delay_seconds=30)
def ingest_task():
    _lazy_imports()
    Ingestor().run()


@task(name="Feature Engineering", retries=1, retry_delay_seconds=30)
def feature_task(tfidf_dim: int = 2000, svd_dim: Optional[int] = 256):
    _lazy_imports()
    FeatureEngineer.run(tfidf_dim=tfidf_dim, svd_dim=svd_dim)


@task(name="Train LGBM", retries=1, retry_delay_seconds=30)
def train_task(trials: int = 30):
    _lazy_imports()
    train(n_trials=trials)


@flow(name="decision_ai_etl_train", log_prints=True)
def decision_ai_flow(tfidf_dim: int = 2000, svd_dim: Optional[int] = 256, trials: int = 30):
    """Pipeline Prefect completo."""
    ingest_task()
    feature_task(tfidf_dim=tfidf_dim, svd_dim=svd_dim)
    train_task(trials=trials)

    if not _PREFECT_AVAILABLE:
        raise RuntimeError(
            "Prefect não está instalado. "
            "Execute `poetry add --group dev prefect` ou use o comando run-local."
        )

# ---------------------------------------------------------------------------
# Execução local sequencial (sem Prefect)
# ---------------------------------------------------------------------------

def run_local(tfidf_dim: int = 2000, svd_dim: Optional[int] = 256, trials: int = 30):
    _lazy_imports()
    logger.info("▶️  Ingest etapa…")
    Ingestor().run()
    logger.info("▶️  Feature Engineering…")
    FeatureEngineer.run(tfidf_dim=tfidf_dim, svd_dim=svd_dim)
    logger.info("▶️  Treinamento…")
    train(n_trials=trials)
    logger.info("🏁 Pipeline local concluído.")


# ---------------------------------------------------------------------------
# Typer CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command("run-prefect")
def run_prefect(
    tfidf_dim: int = 2000,
    svd_dim: Optional[int] = 256,
    trials: int = 30,
):
    """Executa o flow Prefect completo."""
    decision_ai_flow(tfidf_dim=tfidf_dim, svd_dim=svd_dim, trials=trials)


@app.command("run-local")
def run_local_cmd(
    tfidf_dim: int = 2000,
    svd_dim: Optional[int] = 256,
    trials: int = 30,
):
    """Executa pipeline sequencial (sem Prefect)."""
    run_local(tfidf_dim=tfidf_dim, svd_dim=svd_dim, trials=trials)


# ---------------------------------------------------------------------------
# Função utilitária para build de features (para uso externo)
# ---------------------------------------------------------------------------

def build_feature_artifacts(tfidf_dim: int = 2000, svd_dim: Optional[int] = 256):
    """Executa apenas a engenharia de features, para uso por outros módulos."""
    _lazy_imports()
    FeatureEngineer.run(tfidf_dim=tfidf_dim, svd_dim=svd_dim)

if __name__ == "__main__":  # pragma: no cover
    app()
