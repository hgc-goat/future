from __future__ import annotations

import importlib

import pytest

pytest.importorskip("dash")

app_module = importlib.import_module("future.web.app")
data_module = importlib.import_module("future.web.data")


def test_dashboard_data_and_layout(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    data = data_module.load_dashboard_data(root)

    assert not data.ranking.empty
    assert len(data.ranking) == 12
    assert data.ranking["probability"].between(0, 100).all()
    assert data.latest_date is not None

    app = app_module.create_app(data=data)
    assert app.layout is not None
    assert app.title == "future | 板块趋势概率"
