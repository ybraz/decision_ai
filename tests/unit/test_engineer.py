import pandas as pd
import pytest

from decision_ai.features.engineer import FeatureEngineer


def test_fit_transform_raises_on_empty_text():
    """FeatureEngineer.fit_transform should fail when text columns are empty."""
    df = pd.DataFrame({
        "job_text": ["", ""],
        "cv_text": ["", ""],
    })

    fe = FeatureEngineer()
    with pytest.raises(ValueError, match="Coluna job_text vazia"):
        fe.fit_transform(df)


def test_build_dataframe_uses_fallback(monkeypatch):
    """Fallback fields should be used when job description columns are missing."""
    dim_app = pd.DataFrame({"applicant_id": ["1"]})
    dim_job = pd.DataFrame({
        "job_id": ["1"],
        "informacoes_basicas__titulo_vaga": ["Dev"],
    })
    fact = pd.DataFrame({
        "applicant_id": ["1"],
        "job_id": ["1"],
        "status": ["Contratado"],
    })

    monkeypatch.setattr(
        "decision_ai.features.engineer._load_tables",
        lambda: (dim_app, dim_job, fact),
    )

    fe = FeatureEngineer()
    df = fe.build_dataframe()
    assert df.loc[0, "job_text"] == "Dev"


def test_build_dataframe_raises_when_no_description(monkeypatch):
    """Clear error when required job description columns are absent."""
    dim_app = pd.DataFrame({"applicant_id": ["1"]})
    dim_job = pd.DataFrame({"job_id": ["1"]})
    fact = pd.DataFrame({
        "applicant_id": ["1"],
        "job_id": ["1"],
        "status": ["Encaminhado"],
    })

    monkeypatch.setattr(
        "decision_ai.features.engineer._load_tables",
        lambda: (dim_app, dim_job, fact),
    )

    fe = FeatureEngineer()
    with pytest.raises(ValueError, match="Nenhuma coluna de descricao de vaga"):
        fe.build_dataframe()
