from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from future.data.base import Dataset
from future.data.validate import validate_dataset


def test_valid_dataset(small_dataset: Dataset) -> None:
    report = validate_dataset(small_dataset)
    assert report.ok
    assert not report.warnings


def test_missing_column(small_dataset: Dataset) -> None:
    broken = replace(small_dataset, sectors=small_dataset.sectors.drop(columns=["industry_group"]))
    report = validate_dataset(broken)
    assert not report.ok
    assert any("缺少字段" in error for error in report.errors)


def test_duplicate_price_key(small_dataset: Dataset) -> None:
    duplicated = pd.concat(
        [small_dataset.prices, small_dataset.prices.iloc[[0]]],
        ignore_index=True,
    )
    broken = replace(small_dataset, prices=duplicated)
    report = validate_dataset(broken)
    assert not report.ok
    assert any("唯一键" in error for error in report.errors)


def test_bad_ohlc(small_dataset: Dataset) -> None:
    prices = small_dataset.prices.copy()
    prices.loc[prices.index[0], "high"] = 0.0
    report = validate_dataset(replace(small_dataset, prices=prices))
    assert not report.ok
    assert any("OHLC" in error for error in report.errors)


def test_news_sentiment_out_of_range(small_dataset: Dataset) -> None:
    news = small_dataset.news.copy()
    news.loc[news.index[0], "sentiment"] = 2.0
    report = validate_dataset(replace(small_dataset, news=news))
    assert not report.ok
    assert any("sentiment" in error for error in report.errors)


def test_news_count_overflow(small_dataset: Dataset) -> None:
    news = small_dataset.news.copy()
    news.loc[news.index[0], "positive_count"] = news.loc[news.index[0], "news_count"] + 1
    report = validate_dataset(replace(small_dataset, news=news))
    assert not report.ok
    assert any("news_count" in error for error in report.errors)


def test_fundamental_publication_leak(small_dataset: Dataset) -> None:
    fundamentals = small_dataset.fundamentals.copy()
    fundamentals.loc[fundamentals.index[0], "publish_date"] = fundamentals.loc[
        fundamentals.index[0], "period_end"
    ]
    report = validate_dataset(replace(small_dataset, fundamentals=fundamentals))
    assert not report.ok
    assert any("publish_date" in error for error in report.errors)


def test_unknown_sector_id(small_dataset: Dataset) -> None:
    prices = small_dataset.prices.copy()
    prices.loc[prices.index[0], "sector_id"] = "unknown"
    report = validate_dataset(replace(small_dataset, prices=prices))
    assert not report.ok
    assert any("未登记" in error for error in report.errors)


def test_dates_must_be_sorted(small_dataset: Dataset) -> None:
    prices = small_dataset.prices.sample(frac=1.0, random_state=1).reset_index(drop=True)
    report = validate_dataset(replace(small_dataset, prices=prices))
    assert not report.ok
    assert any("升序" in error for error in report.errors)


@pytest.mark.parametrize("column", ["open", "close", "volume", "amount"])
def test_negative_price_fields(small_dataset: Dataset, column: str) -> None:
    prices = small_dataset.prices.copy()
    prices.loc[prices.index[0], column] = -1.0
    report = validate_dataset(replace(small_dataset, prices=prices))
    assert not report.ok
    assert any(column in error for error in report.errors)
