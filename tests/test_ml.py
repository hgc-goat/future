from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from future.data.base import DataSourceError
from future.factors import compute_factors
from future.ml import (
    FEATURE_COLUMNS,
    ModelConfig,
    train_advanced_model,
    write_advanced_outputs,
)
from future.news import compute_news_fusion

pytest.importorskip("sklearn")


MODEL_CONFIG = ModelConfig(
    train_start=date(2022, 1, 1),
    train_end=date(2023, 12, 31),
    oos_start=date(2024, 1, 1),
    oos_end=date(2025, 12, 31),
    max_iter=40,
    min_samples_leaf=20,
)


@pytest.fixture(scope="module")
def fused(stage2_dataset):
    factors = compute_factors(stage2_dataset)
    return compute_news_fusion(factors, stage2_dataset.news)


@pytest.fixture(scope="module")
def result(stage2_dataset, fused):
    return train_advanced_model(stage2_dataset, fused, MODEL_CONFIG)


def test_model_metrics_schema(result) -> None:
    metrics = result.metrics

    assert metrics["model"]["class"] == "HistGradientBoostingClassifier"
    assert metrics["model"]["features"] == list(FEATURE_COLUMNS)
    assert metrics["train_rows"] > 0
    assert metrics["oos_rows"] > 0
    assert metrics["train_label_end"] <= "2023-12-31"
    assert set(metrics["baseline"]) >= {"long_short", "oos_ic", "layers"}
    assert set(metrics["hgb"]) >= {"long_short", "oos_ic", "layers"}
    assert set(metrics["delta"]) == {
        "ic_5",
        "ic_20",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
    }


def test_predictions_range(result) -> None:
    predictions = result.predictions

    assert list(predictions.columns) == ["date", "sector_id", "p_up"]
    assert predictions["p_up"].notna().all()
    assert predictions["p_up"].between(0.0, 1.0).all()
    assert not predictions.duplicated(subset=["date", "sector_id"]).any()


def test_model_reproducible(stage2_dataset, fused) -> None:
    first = train_advanced_model(stage2_dataset, fused, MODEL_CONFIG)
    second = train_advanced_model(stage2_dataset, fused, MODEL_CONFIG)

    assert first.metrics == second.metrics
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


def test_missing_feature_raises(stage2_dataset, fused) -> None:
    broken = fused.drop(columns=["news_confirm"])
    with pytest.raises(DataSourceError, match="特征缺少"):
        train_advanced_model(stage2_dataset, broken, MODEL_CONFIG)


def test_model_config_validation() -> None:
    with pytest.raises(ValueError, match="horizon"):
        ModelConfig(horizon=0)
    with pytest.raises(ValueError, match="train_start"):
        ModelConfig(
            train_start=date(2024, 1, 1),
            train_end=date(2023, 1, 1),
        )


def test_write_advanced_outputs(tmp_path, result) -> None:
    manifest = write_advanced_outputs(result, tmp_path)

    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert manifest["files"]["predictions.csv"]["rows"] == len(result.predictions)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["train_rows"] == result.metrics["train_rows"]
