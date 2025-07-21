from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import logging
import sys

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


def _sha256(value: str | None) -> str | None:
    """Return SHA-256 hex digest for non-empty strings."""
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Ingestor:
    """Handle conversion from raw JSON files to Parquet tables."""

    def __init__(self, raw_dir: Path, out_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)

    # ------------------------------------------------------------------
    def _load_json(self, name: str) -> Dict[str, Any]:
        path = self.raw_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("%s carregado (%d registros)", name, len(data))
        return data

    def _save_parquet(self, df: pd.DataFrame, name: str) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / name
        df.to_parquet(path, index=False)
        try:
            subprocess.run(["dvc", "add", str(path)], check=False)
        except Exception:
            pass
        logger.info("%s salvo (%d linhas)", name, len(df))

    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("Iniciando ingestão de dados…")
        apps = self._load_json("applicants.json")
        jobs = self._load_json("vagas.json")
        prospects = self._load_json("prospects.json")

        self._process_applicants(apps)
        self._process_jobs(jobs)
        self._process_prospects(prospects)

    # ------------------------------------------------------------------
    def _process_applicants(self, data: Dict[str, Any]) -> None:
        logger.info("Processando candidatos…")
        records = []
        for app_id, sections in data.items():
            info = sections.get("infos_basicas", {})
            record = {
                "applicant_id": int(info.get("codigo_profissional", app_id)),
                "infos_basicas__telefone": _sha256(info.get("telefone")),
                "infos_basicas__email": _sha256(info.get("email")),
                "infos_basicas__nome": info.get("nome"),
            }
            cv_pt = sections.get("cv_pt") or sections.get("cv")
            if cv_pt:
                record["cv_pt"] = cv_pt
            if sections.get("cv_en"):
                record["cv_en"] = sections["cv_en"]
            records.append(record)
        df = pd.DataFrame(records)
        self._save_parquet(df, "dim_applicant.parquet")

    def _process_jobs(self, data: Dict[str, Any]) -> None:
        logger.info("Processando vagas…")
        records = []
        for job_id, obj in data.items():
            info = obj.get("informacoes_basicas", {})
            perfil = obj.get("perfil_vaga", {})
            record = {"job_id": int(job_id)}
            for k, v in info.items():
                record[f"informacoes_basicas__{k}"] = v
            for k, v in perfil.items():
                record[f"perfil__{k}"] = v
            records.append(record)
        df = pd.DataFrame(records)
        self._save_parquet(df, "dim_job.parquet")

    def _process_prospects(self, data: Dict[str, Any]) -> None:
        logger.info("Processando prospects…")
        records = []
        for job_id, obj in data.items():
            for prospect in obj.get("prospects", []):
                record = {
                    "job_id": int(job_id),
                    "applicant_id": int(prospect.get("codigo")),
                    "status": prospect.get("situacao_candidado"),
                }
                records.append(record)
        df = pd.DataFrame(records)
        self._save_parquet(df, "fact_prospect.parquet")


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run data ingestion pipeline")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    ing = Ingestor(raw_dir=args.raw_dir, out_dir=args.out_dir)
    ing.run()


if __name__ == "__main__":  # pragma: no cover
    main()
