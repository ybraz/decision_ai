"""Unit tests for decision_ai.data.ingest module.

These tests use pytest and operate entirely inside a temporary
filesystem (pytest's *tmp_path* fixture) to avoid any side‑effects on
real project data. External calls to **DVC** (subprocess.run) are
monkey‑patched so the suite can be executed on CI environments where
DVC may be unavailable.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from decision_ai.data.ingest import Ingestor, _sha256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_sha256_basic() -> None:
    """_sha256 must hash non‑empty strings and keep None/empty unchanged."""
    assert _sha256("123") == hashlib.sha256(b"123").hexdigest()
    assert _sha256("") is None
    assert _sha256(None) is None


@pytest.mark.parametrize("missing_file", ["applicants.json", "prospects.json", "vagas.json"])
def test_ingestor_missing_files(tmp_path: Path, missing_file: str) -> None:
    """When a required JSON file is absent, Ingestor should raise FileNotFoundError."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # create the other two minimal files so only one is missing
    minimal_applicant = {
        "1": {
            "infos_basicas": {
                "telefone": "(11) 91234‑5678",
                "email": "foo@example.com",
                "codigo_profissional": "1",
                "nome": "Foo Bar",
            }
        }
    }
    minimal_prospect = {"100": {"titulo": "Job", "prospects": []}}
    minimal_job = {
        "100": {
            "informacoes_basicas": {"titulo_vaga": "Dev"},
            "perfil_vaga": {},
        }
    }
    if missing_file != "applicants.json":
        _write_json(raw_dir / "applicants.json", minimal_applicant)
    if missing_file != "prospects.json":
        _write_json(raw_dir / "prospects.json", minimal_prospect)
    if missing_file != "vagas.json":
        _write_json(raw_dir / "vagas.json", minimal_job)

    ingestor = Ingestor(raw_dir=raw_dir, out_dir=tmp_path / "processed")
    with pytest.raises(FileNotFoundError):
        ingestor.run()


def test_ingestor_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End‑to‑end ingestion produces all Parquet tables with hashed PII."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(); processed_dir.mkdir()

    # ---------------- create minimal but valid datasets ----------------
    applicants = {
        "42": {
            "infos_basicas": {
                "telefone": "(11) 90000‑0000",
                "email": "alice@test.com",
                "codigo_profissional": "42",
                "nome": "Alice",
            }
        }
    }
    prospects = {
        "777": {
            "titulo": "Data Engineer",
            "prospects": [
                {
                    "nome": "Alice",
                    "codigo": "42",
                    "situacao_candidado": "Encaminhado",
                    "data_candidatura": "01‑01‑2025",
                    "ultima_atualizacao": "02‑01‑2025",
                    "comentario": "",
                    "recrutador": "Bob",
                }
            ],
        }
    }
    vagas = {
        "777": {
            "informacoes_basicas": {"titulo_vaga": "Data Engineer"},
            "perfil_vaga": {"nivel_profissional": "Pleno"},
        }
    }
    _write_json(raw_dir / "applicants.json", applicants)
    _write_json(raw_dir / "prospects.json", prospects)
    _write_json(raw_dir / "vagas.json", vagas)

    # ---------------- monkey‑patch subprocess.run so DVC isn't required --
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    # ---------------- run pipeline ------------------------------------
    ingestor = Ingestor(raw_dir=raw_dir, out_dir=processed_dir)
    ingestor.run()

    # ---------------- assertions --------------------------------------
    dim_app = pd.read_parquet(processed_dir / "dim_applicant.parquet")
    dim_job = pd.read_parquet(processed_dir / "dim_job.parquet")
    fact = pd.read_parquet(processed_dir / "fact_prospect.parquet")

    # table shapes
    assert dim_app.shape[0] == 1
    assert dim_job.shape[0] == 1
    assert fact.shape[0] == 1

    # check hashing (64‑char hex) & not equal original
    hashed_phone = dim_app.loc[0, "infos_basicas__telefone"]
    assert hashed_phone != "(11) 90000‑0000"
    assert len(hashed_phone) == 64

    hashed_email = dim_app.loc[0, "infos_basicas__email"]
    assert hashed_email != "alice@test.com"
    assert len(hashed_email) == 64

    # referential integrity
    assert fact.loc[0, "applicant_id"] == 42
    assert fact.loc[0, "job_id"] == 777

    # column naming convention for job profile fields
    assert "profile__nivel_profissional" in dim_job.columns
