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
