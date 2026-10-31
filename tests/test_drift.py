from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from future.data.base import DataSourceError
from future.drift import (
    DriftConfig,
    compute_drift,
    population_stability_index,
    write_drift_outputs,
)
from future.factors import compute_factors
from future.news import compute_news_fusion

DRIFT_CONFIG = DriftConfig(
    train_start=date(2022, 1, 1),
    train_end=date(2023, 12, 31),
    oos_start=date(2024, 1, 1),
    oos_end=date(2025, 12, 31),
)


@pytest.fixture(scope="module")
def fused(stage2_dataset):
    factors = compute_factors(stage2_dataset)
    return compute_news_fusion(factors, stage2_dataset.news)


@pytest.fixture(scope="module")
def drift_result(stage2_dataset, fused):
    return compute_drift(stage2_dataset, fused, DRIFT_CONFIG)


def test_psi_identical_is_zero() -> None:
    values = pd.Series(np.linspace(-1.0, 1.0, 500))
    assert population_stability_index(values, values) < 1e-9


def test_psi_shifted_is_positive() -> None:
    expected = pd.Series(np.random.default_rng(1).normal(0.0, 1.0, 500))
    actual = pd.Series(np.random.default_rng(2).normal(2.0, 1.0, 500))
    assert population_stability_index(expected, actual) > 0.1


def test_drift_metrics_schema(drift_result) -> None:
    metrics = drift_result.metrics

    assert set(metrics["train"]) == {"ic_mean", "ic_std", "n"}
    assert set(metrics["latest"]) == {"date", "rolling_ic", "ic"}
    assert set(metrics["psi"]) == {"by_feature", "max", "average"}
    assert set(metrics["severity"]) == {"ic", "distribution", "overall", "level"}
    assert metrics["severity"]["level"] in {"low", "medium", "high"}
    assert 0.0 <= metrics["severity"]["overall"] <= 1.0
    assert 0.0 <= metrics["severity"]["ic"] <= 1.0
    assert 0.0 <= metrics["severity"]["distribution"] <= 1.0
    assert metrics["psi"]["max"] >= metrics["psi"]["average"] >= 0.0
    assert not drift_result.rolling_ic.empty
    assert list(drift_result.rolling_ic.columns) == ["date", "ic", "rolling_ic"]


def test_drift_reproducible(stage2_dataset, fused) -> None:
    first = compute_drift(stage2_dataset, fused, DRIFT_CONFIG)
    second = compute_drift(stage2_dataset, fused, DRIFT_CONFIG)

    assert first.metrics == second.metrics
    pd.testing.assert_frame_equal(first.rolling_ic, second.rolling_ic)


def test_drift_requires_score(stage2_dataset, fused) -> None:
    config = DriftConfig(
        train_start=date(2022, 1, 1),
        train_end=date(2023, 12, 31),
        oos_start=date(2024, 1, 1),
        oos_end=date(2025, 12, 31),
        score_column="missing",
    )
    with pytest.raises(DataSourceError, match="missing"):
        compute_drift(stage2_dataset, fused, config)


def test_drift_config_validation() -> None:
    with pytest.raises(ValueError, match="horizon"):
        DriftConfig(horizon=0)
    with pytest.raises(ValueError, match="psi"):
        DriftConfig(psi_bins=1)


def test_write_drift_outputs(tmp_path, drift_result) -> None:
    manifest = write_drift_outputs(drift_result, tmp_path)

    assert (tmp_path / "drift.json").is_file()
    assert (tmp_path / "rolling_ic.csv").is_file()
    assert manifest["files"]["rolling_ic.csv"]["rows"] == len(drift_result.rolling_ic)
    metrics = json.loads((tmp_path / "drift.json").read_text(encoding="utf-8"))
    assert metrics["severity"] == drift_result.metrics["severity"]
