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
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import CrossEncoder
import math

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

# Modelo de similaridade semântica para cv_text × job_text
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ApplicantIn(BaseModel):
    cv_text: str = Field(..., description="Conteúdo do currículo")
    job_text: str | None = Field("", description="Descrição da vaga (opcional)")

    # allow any extra keys to pass through to the DataFrame
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
    # Extrai pares de texto
    pairs = [[item.cv_text, item.job_text or ""] for item in payload]
    # Previsão de similaridade
    scores = cross_encoder.predict(pairs)
    # Converte scores em probabilidade via sigmoid e formata saída
    results = []
    for score in scores:
        prob = 1.0 / (1.0 + math.exp(-float(score)))
        results.append({"proba": prob, "pred": int(prob >= THRESHOLD)})
    return results


# ---------------------------------------------------------------------------
# Health & Metadata
# ---------------------------------------------------------------------------


@app.get("/healthz", summary="Liveness probe")
def health():
    return {"status": "ok", "model_loaded": False}


# ---------------------------------------------------------------------------
# Custom exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Error: {type(exc).__name__}"},
    )
