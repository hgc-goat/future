from __future__ import annotations

from future.start import artifacts_ready


def test_artifacts_ready(pipeline_artifacts) -> None:
    root, _ = pipeline_artifacts
    assert artifacts_ready(root)


def test_artifacts_not_ready(tmp_path) -> None:
    assert not artifacts_ready(tmp_path)
