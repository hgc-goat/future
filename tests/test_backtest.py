from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from future.backtest import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
    write_backtest,
)
from future.factors import compute_factors


@pytest.fixture(scope="module")
def stage2_config() -> BacktestConfig:
    return BacktestConfig(
        train_start=date(2022, 1, 1),
        train_end=date(2023, 12, 31),
        oos_start=date(2024, 1, 1),
        oos_end=date(2025, 12, 31),
        n_layers=5,
        min_sectors=5,
    )


@pytest.fixture(scope="module")
def stage2_factors(stage2_dataset):
    return compute_factors(stage2_dataset)


@pytest.fixture(scope="module")
def stage2_result(stage2_dataset, stage2_factors, stage2_config):
    return run_backtest(stage2_dataset, stage2_factors, stage2_config)


def test_metrics_schema(stage2_result: BacktestResult) -> None:
    metrics = stage2_result.metrics

    assert set(metrics["layers"]) == {"1", "2", "3", "4", "5"}
    assert metrics["long_short"]["n_days"] > 0
    assert metrics["n_dates"] > 0
    assert metrics["n_rebalances"] >= 2
    assert set(metrics["oos_ic"]) == {
        "momentum",
        "financial",
        "flow",
        "crowding",
        "composite",
    }
    assert set(metrics["oos_ic"]["composite"]) == {"5", "20"}
    assert metrics["long_short"]["annualized_return"] is not None
    assert metrics["long_short"]["max_drawdown"] <= 0


def test_layer_returns(stage2_result: BacktestResult) -> None:
    returns = stage2_result.layer_returns

    assert list(returns.columns) == [
        "layer_1",
        "layer_2",
        "layer_3",
        "layer_4",
        "layer_5",
        "long_short",
    ]
    assert not returns.isna().any().any()
    pd.testing.assert_series_equal(
        returns["long_short"],
        returns["layer_5"] - returns["layer_1"],
        check_names=False,
    )


def test_ic_series_is_oos(stage2_result: BacktestResult, stage2_config: BacktestConfig) -> None:
    ic_series = stage2_result.ic_series

    assert list(ic_series.columns) == ["date", "factor", "horizon", "ic"]
    assert not ic_series.empty
    assert ic_series["date"].min().date() >= stage2_config.oos_start
    assert ic_series["date"].max().date() <= stage2_config.oos_end


def test_backtest_reproducible(stage2_dataset, stage2_factors, stage2_config) -> None:
    first = run_backtest(stage2_dataset, stage2_factors, stage2_config)
    second = run_backtest(stage2_dataset, stage2_factors, stage2_config)

    assert first.metrics == second.metrics
    pd.testing.assert_frame_equal(first.layer_returns, second.layer_returns)
    pd.testing.assert_frame_equal(first.ic_series, second.ic_series)


def test_write_backtest(tmp_path, stage2_result: BacktestResult) -> None:
    manifest = write_backtest(stage2_result, tmp_path)

    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "layer_returns.csv").is_file()
    assert (tmp_path / "ic_series.csv").is_file()
    assert manifest["files"]["layer_returns.csv"]["rows"] == len(stage2_result.layer_returns)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["long_short"]["n_days"] == stage2_result.metrics["long_short"]["n_days"]


def test_invalid_backtest_config() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        BacktestConfig(n_layers=1)

    with pytest.raises(ValueError, match="min_sectors"):
        BacktestConfig(n_layers=5, min_sectors=4)

    with pytest.raises(ValueError, match="train_start"):
        BacktestConfig(
            train_start=date(2024, 1, 1),
            train_end=date(2023, 1, 1),
        )


def test_backtest_requires_factors(stage2_dataset) -> None:
    from future.data.base import DataSourceError

    with pytest.raises(DataSourceError, match="缺少字段"):
        run_backtest(stage2_dataset, pd.DataFrame({"date": [], "sector_id": []}))
