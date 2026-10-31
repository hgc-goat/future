from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from future.data.base import Dataset
from future.data.mock import generate_dataset
from future.data.validate import validate_dataset

SMALL_PARAMS = {
    "seed": 7,
    "start": date(2024, 1, 1),
    "end": date(2024, 6, 30),
    "sector_limit": 6,
}


def test_small_dataset_is_valid(small_dataset: Dataset) -> None:
    report = validate_dataset(small_dataset)
    assert report.ok, report.errors
    assert len(small_dataset.sectors) == 6
    assert len(small_dataset.prices) == 6 * 130
    assert len(small_dataset.fundamentals) == 6 * 2
    assert len(small_dataset.flows) == len(small_dataset.prices)
    assert len(small_dataset.news) == len(small_dataset.prices)


def test_same_seed_is_reproducible() -> None:
    first = generate_dataset(**SMALL_PARAMS)
    second = generate_dataset(**SMALL_PARAMS)

    for name in ("sectors", "prices", "fundamentals", "flows", "news"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))


def test_different_seed_changes_prices() -> None:
    first = generate_dataset(**SMALL_PARAMS)
    second = generate_dataset(**{**SMALL_PARAMS, "seed": 8})
    assert not first.prices["close"].equals(second.prices["close"])


def test_fundamentals_have_publication_lag(small_dataset: Dataset) -> None:
    lag = small_dataset.fundamentals["publish_date"] - small_dataset.fundamentals["period_end"]
    assert (lag >= pd.Timedelta(days=30)).all()
    assert (
        small_dataset.fundamentals["publish_date"] > small_dataset.fundamentals["period_end"]
    ).all()


def test_daily_tables_share_dates(small_dataset: Dataset) -> None:
    price_dates = set(small_dataset.prices["date"])
    assert set(small_dataset.flows["date"]) == price_dates
    assert set(small_dataset.news["date"]) == price_dates


def test_ohlc_and_news_consistency(small_dataset: Dataset) -> None:
    prices = small_dataset.prices
    body_high = prices[["open", "close"]].max(axis=1)
    body_low = prices[["open", "close"]].min(axis=1)
    assert (prices["high"] >= body_high).all()
    assert (prices["low"] <= body_low).all()

    news = small_dataset.news
    assert (news["positive_count"] + news["negative_count"] <= news["news_count"]).all()
    assert news["sentiment"].between(-1.0, 1.0).all()


def test_metadata(small_dataset: Dataset) -> None:
    assert small_dataset.metadata["seed"] == 7
    assert small_dataset.metadata["source"] == "mock"
    assert small_dataset.metadata["n_sectors"] == 6
    assert small_dataset.metadata["n_days"] == 130


def test_invalid_sector_limit() -> None:
    with pytest.raises(ValueError, match="sector_limit"):
        generate_dataset(sector_limit=0)


def test_too_short_range() -> None:
    with pytest.raises(ValueError, match="交易日"):
        generate_dataset(start=date(2024, 1, 1), end=date(2024, 1, 2))
