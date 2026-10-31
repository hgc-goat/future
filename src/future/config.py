"""运行配置。

所有默认值都可以通过环境变量覆盖；这里不读取、不保存任何密钥。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DEFAULT_START = date(2020, 1, 2)
DEFAULT_END = date(2025, 12, 31)
DEFAULT_SEED = 42
DEFAULT_DATA_DIR = Path("data/mock")
DEFAULT_SOURCE = "mock"
SUPPORTED_SOURCES = ("mock", "csv", "real")

_ENV_KEYS = (
    "FUTURE_SEED",
    "FUTURE_START",
    "FUTURE_END",
    "FUTURE_DATA_SOURCE",
    "FUTURE_DATA_DIR",
)


def _parse_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 YYYY-MM-DD 格式，当前值：{value!r}") from exc


def _parse_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise ValueError(f"FUTURE_SEED 必须是整数，当前值：{value!r}") from exc
    if seed < 0:
        raise ValueError("FUTURE_SEED 必须是非负整数")
    return seed


@dataclass(frozen=True)
class Settings:
    """应用运行参数。"""

    seed: int = DEFAULT_SEED
    start: date = DEFAULT_START
    end: date = DEFAULT_END
    data_source: str = DEFAULT_SOURCE
    data_dir: Path = DEFAULT_DATA_DIR

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("start 必须早于 end")
        if self.seed < 0:
            raise ValueError("seed 必须是非负整数")
        if self.data_source not in SUPPORTED_SOURCES:
            raise ValueError(f"不支持的 data_source：{self.data_source!r}")

    @classmethod
    def from_env(cls) -> Settings:
        """从 FUTURE_* 环境变量构造配置，未设置时使用默认值。"""

        source = os.getenv("FUTURE_DATA_SOURCE", DEFAULT_SOURCE).strip().lower()
        return cls(
            seed=_parse_seed(os.getenv("FUTURE_SEED", str(DEFAULT_SEED))),
            start=_parse_date(os.getenv("FUTURE_START", DEFAULT_START.isoformat()), "FUTURE_START"),
            end=_parse_date(os.getenv("FUTURE_END", DEFAULT_END.isoformat()), "FUTURE_END"),
            data_source=source,
            data_dir=Path(os.getenv("FUTURE_DATA_DIR", str(DEFAULT_DATA_DIR))),
        )


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_END",
    "DEFAULT_SEED",
    "DEFAULT_SOURCE",
    "DEFAULT_START",
    "SUPPORTED_SOURCES",
    "Settings",
]
