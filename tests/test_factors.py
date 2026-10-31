from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from future.data.base import Dataset, DataSourceError
from future.factors import (
    FACTOR_COLUMNS,
    FactorConfig,
    compute_factors,
    read_factors,
    write_factors,
)


def test_factor_schema(stage2_dataset: Dataset) -> None:
    factors = compute_factors(stage2_dataset)

    assert list(factors.columns) == list(FACTOR_COLUMNS)
    assert not factors.duplicated(subset=["date", "sector_id"]).any()
    assert factors["composite"].notna().all()
    assert factors["date"].min() >= pd.Timestamp("2022-01-01")
    assert factors["date"].max() <= pd.Timestamp("2025-12-31")


def test_factor_reproducible(stage2_dataset: Dataset) -> None:
    first = compute_factors(stage2_dataset)
    second = compute_factors(stage2_dataset)
    pd.testing.assert_frame_equal(first, second)


def test_factor_no_lookahead(stage2_dataset: Dataset) -> None:
    cutoff = pd.Timestamp("2023-06-30")

    prices = stage2_dataset.prices.copy()
    future_price = prices["date"] > cutoff
    for column in ("open", "high", "low", "close"):
        prices.loc[future_price, column] *= 3.0
    for column in ("volume", "amount"):
        prices.loc[future_price, column] *= 2.0

    flows = stage2_dataset.flows.copy()
    future_flow = flows["date"] > cutoff
    flows.loc[future_flow, "main_net_inflow"] *= 5.0
    flows.loc[future_flow, "northbound_net"] *= 5.0
    flows.loc[future_flow, "crowding_raw"] = 1.0 - flows.loc[future_flow, "crowding_raw"]

    fundamentals = stage2_dataset.fundamentals.copy()
    future_fund = fundamentals["publish_date"] > cutoff
    fundamentals.loc[future_fund, "roe_ttm"] += 30.0
    fundamentals.loc[future_fund, "revenue_yoy"] += 0.5
    fundamentals.loc[future_fund, "profit_yoy"] += 0.8

    modified = replace(
        stage2_dataset,
        prices=prices,
        flows=flows,
        fundamentals=fundamentals,
    )
    baseline = compute_factors(stage2_dataset)
    changed = compute_factors(modified)

    baseline_prefix = baseline.loc[baseline["date"] <= cutoff].reset_index(drop=True)
    changed_prefix = changed.loc[changed["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_prefix, changed_prefix)


def test_crowding_is_reverse_hint(stage2_dataset: Dataset) -> None:
    factors = compute_factors(stage2_dataset)
    config = FactorConfig()
    expected = (
        factors[["momentum", "financial", "flow"]].mean(axis=1)
        - config.crowding_weight * factors["crowding"]
    )
    pd.testing.assert_series_equal(
        factors["composite"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_write_and_read_factors(tmp_path, stage2_dataset: Dataset) -> None:
    factors = compute_factors(stage2_dataset)
    manifest = write_factors(factors, tmp_path)
    loaded = read_factors(tmp_path)

    assert manifest["rows"] == len(factors)
    assert (tmp_path / "factors.csv").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert list(loaded.columns) == list(FACTOR_COLUMNS)
    assert len(loaded) == len(factors)


def test_read_factors_missing(tmp_path) -> None:
    with pytest.raises(DataSourceError, match="不存在"):
        read_factors(tmp_path / "missing.csv")
