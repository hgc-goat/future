from __future__ import annotations

from datetime import date

import pytest

from future.data.base import Dataset
from future.data.mock import generate_dataset
from future.pipeline import PipelineConfig, run_pipeline

ENV_KEYS = (
    "FUTURE_SEED",
    "FUTURE_START",
    "FUTURE_END",
    "FUTURE_DATA_SOURCE",
    "FUTURE_DATA_DIR",
)


@pytest.fixture(autouse=True)
def clear_future_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def small_dataset() -> Dataset:
    return generate_dataset(
        seed=7,
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        sector_limit=6,
    )


@pytest.fixture(scope="session")
def stage2_dataset() -> Dataset:
    return generate_dataset(
        seed=11,
        start=date(2022, 1, 1),
        end=date(2025, 12, 31),
        sector_limit=12,
    )


@pytest.fixture(scope="session")
def pipeline_artifacts(tmp_path_factory) -> tuple:
    """在临时目录从零跑完整流水线，供端到端测试复用。"""

    root = tmp_path_factory.mktemp("pipeline")
    result = run_pipeline(
        PipelineConfig(
            root=root,
            seed=7,
            start=date(2022, 1, 1),
            end=date(2024, 12, 31),
            train_start=date(2022, 1, 1),
            train_end=date(2023, 12, 31),
            oos_start=date(2024, 1, 1),
            oos_end=date(2024, 12, 31),
            factor_start=date(2022, 1, 1),
            sector_limit=12,
        )
    )
    return root, result
