"""
decision_ai.api.main
====================

REST API para servir o modelo de classificação de candidatos.

Recursos
--------
* **POST /predict**  
  Recebe JSON ‑ lista de objetos ou objeto único com pelo menos `cv_text`
  (opcional `job_text`).  
  Devolve JSON com `proba` e `pred` (threshold configurável).

Segurança
---------
* Autenticação **API‑Key** simples via header `X-API-Key`.
* Carrega artefatos na inicialização (singleton).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field, validator, ConfigDict
from functools import lru_cache
from decision_ai.models.predict import load_latest_artifacts, predict as _predict_df

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("DECISION_AI_API_KEY", "dev‑secret")  # override em prod
THRESHOLD = float(os.getenv("DECISION_AI_THRESHOLD", "0.25"))

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Decision AI Prediction API",
    version="0.1.0",
    description="Serviço de inferência para classificação de candidatos.",
)

# Lazy loading of sklearn pipeline and model
@lru_cache(maxsize=1)
def _get_artifacts():
    return load_latest_artifacts()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ApplicantIn(BaseModel):
    cv_text: str = Field(..., description="Conteúdo (pt+en) do currículo")
    job_text: str | None = Field("", description="Descrição da vaga (opcional)")
    base__tipo_contratacao: str | None = ""
    profile__nivel_profissional: str | None = ""
    profile__nivel_academico: str | None = ""
    profile__nivel_ingles: str | None = ""
    informacoes_profissionais__remuneracao: float | int | None = 0

    @validator("cv_text", "job_text", pre=True, always=True)
    def _to_str(cls, v):
        return "" if v is None else str(v)

    model_config = ConfigDict(extra="allow")


class PredictionOut(BaseModel):
    proba: float = Field(..., ge=0.0, le=1.0)
    pred: int = Field(..., ge=0, le=1)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def require_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/predict",
    response_model=List[PredictionOut],
    summary="Classify applicants",
    dependencies=[Depends(require_api_key)],
)
def predict_endpoint(payload: List[ApplicantIn]):
    if not payload:
        raise HTTPException(status_code=400, detail="Empty payload")
    df = pd.DataFrame(
        [
            item.model_dump(mode="python", exclude_none=False)
            for item in payload
        ]
    )
    _model, _pipeline = _get_artifacts()
    preds = _predict_df(df, threshold=THRESHOLD)
    return [PredictionOut(proba=float(p), pred=int(y)) for p, y in zip(preds["proba"], preds["pred"])]


# ---------------------------------------------------------------------------
# Health & Metadata
# ---------------------------------------------------------------------------


@app.get("/healthz", summary="Liveness probe")
def health():
    return {"status": "ok", "model_loaded": _get_artifacts.cache_info().hits > 0}


# ---------------------------------------------------------------------------
# Custom exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Error: {type(exc).__name__}"},
    )
