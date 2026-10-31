from __future__ import annotations

import pytest

from future.web.deepseek_bridge import DeepSeekBridgeError, _find_textbox, ask_deepseek


def test_find_textbox_from_observation() -> None:
    payload = {"observation": {"text": '@e1 button\n@e5 textbox "给 DeepSeek 发送消息" [empty]'}}
    assert _find_textbox(payload) == "@e5"


def test_ask_rejects_empty_prompt() -> None:
    with pytest.raises(DeepSeekBridgeError, match="提示词为空"):
        ask_deepseek("   ")


def test_ask_reports_missing_bsk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("future.web.deepseek_bridge.shutil.which", lambda _name: None)
    with pytest.raises(DeepSeekBridgeError, match="未找到 bsk"):
        ask_deepseek("测试")
