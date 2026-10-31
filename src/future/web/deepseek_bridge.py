"""实验性本地 DeepSeek 网页版桥接。

只通过 bsk 操作已经登录的浏览器页面，不读取 Cookie、密码或登录 token。
网页版不是稳定 API；失败时调用方应回退到“复制提示词 + 打开网页版”。
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

DEEPSEEK_URL = "https://chat.deepseek.com/"
ANSWER_SELECTOR = ".ds-assistant-message-main-content"
MAX_WAIT_SECONDS = 180
STABLE_POLLS = 3


class DeepSeekBridgeError(RuntimeError):
    """网页版不可用、未登录或自动化失败。"""


@dataclass(frozen=True)
class DeepSeekAnswer:
    answer: str
    duration_seconds: float


def _bsk() -> str:
    executable = shutil.which("bsk")
    if not executable:
        raise DeepSeekBridgeError("未找到 bsk CLI，无法使用本机浏览器自动化")
    return executable


def _run(args: list[str], timeout: int = 30) -> str:
    try:
        completed = subprocess.run(
            [_bsk(), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DeepSeekBridgeError(f"bsk {args[0]} 超时") from exc
    except OSError as exc:
        raise DeepSeekBridgeError(f"bsk 启动失败：{exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise DeepSeekBridgeError(f"bsk {args[0]} 失败：{detail[:240]}")
    return completed.stdout.strip()


def _parse_json(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeepSeekBridgeError(f"无法解析 bsk 输出：{text[:200]}") from exc
    if not isinstance(value, dict):
        raise DeepSeekBridgeError("bsk 返回格式不是对象")
    return value


def _sync(session_id: str) -> dict:
    last_error = "页面未就绪"
    for _ in range(8):
        try:
            payload = _parse_json(
                _run(["cooperate", "sync", "--session", session_id, "--json"], timeout=20)
            )
            if payload.get("context_id") and payload.get("kind") in {
                "baseline",
                "unchanged",
            }:
                return payload
            last_error = (
                f"页面同步未完成：{payload.get('code') or ''} {payload.get('message') or ''}"
            ).strip()
        except DeepSeekBridgeError as exc:
            last_error = str(exc)
        with contextlib.suppress(DeepSeekBridgeError):
            _run(["cooperate", "wait", "--session", session_id], timeout=15)
        time.sleep(1.5)
    raise DeepSeekBridgeError(last_error)


def _find_textbox(sync_payload: dict) -> str:
    text = sync_payload.get("observation", {}).get("text", "")
    for line in text.splitlines():
        if "textbox" not in line:
            continue
        if "给 DeepSeek 发送消息" not in line and "发送消息" not in line:
            continue
        match = re.search(r"@e\d+", line)
        if match:
            return match.group(0)
    raise DeepSeekBridgeError("未找到 DeepSeek 输入框，请先在浏览器登录网页版")


def _answer_state(session_id: str, context_id: str) -> tuple[int, str]:
    expression = (
        "(() => {"
        f"const nodes = document.querySelectorAll('{ANSWER_SELECTOR}');"
        "const last = nodes[nodes.length - 1];"
        "return JSON.stringify({count: nodes.length, text: last ? last.innerText : ''});"
        "})()"
    )
    raw = _run(
        ["evaluate", expression, "--session", session_id, "--context", context_id],
        timeout=25,
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DeepSeekBridgeError("无法读取 DeepSeek 回答") from exc
    return int(payload.get("count", 0)), str(payload.get("text", ""))


def _wait_for_answer(session_id: str, initial_count: int) -> str:
    started = time.monotonic()
    last_text = ""
    stable = 0
    while time.monotonic() - started < MAX_WAIT_SECONDS:
        time.sleep(2)
        try:
            sync_payload = _sync(session_id)
            count, text = _answer_state(session_id, str(sync_payload["context_id"]))
        except DeepSeekBridgeError:
            # 页面流式生成时上下文会变化，等待下一轮重新同步。
            continue
        if count > initial_count and text:
            if text == last_text:
                stable += 1
            else:
                stable = 0
            last_text = text
            if stable >= STABLE_POLLS:
                return last_text
    if last_text:
        return last_text
    raise DeepSeekBridgeError("等待 DeepSeek 回答超时；请检查网页版是否可用")


def ask_deepseek(prompt: str) -> DeepSeekAnswer:
    """打开已验证登录的网页版，发送提示词并读取最后一条回答。"""

    if not prompt.strip():
        raise DeepSeekBridgeError("提示词为空")
    session_id: str | None = None
    started = time.monotonic()
    try:
        session_payload = _parse_json(_run(["session", "start", "--json"], timeout=25))
        session_id = str(session_payload["session_id"])
        _run(["navigate", DEEPSEEK_URL, "--session", session_id], timeout=45)
        time.sleep(3)

        sync_payload = _sync(session_id)
        textbox = _find_textbox(sync_payload)
        _run(
            [
                "fill",
                textbox,
                "--value",
                prompt,
                "--session",
                session_id,
                "--context",
                str(sync_payload["context_id"]),
            ],
            timeout=30,
        )

        sync_payload = _sync(session_id)
        textbox = _find_textbox(sync_payload) or textbox
        _run(
            [
                "press",
                "Enter",
                "--ref",
                textbox,
                "--session",
                session_id,
                "--context",
                str(sync_payload["context_id"]),
            ],
            timeout=20,
        )
        answer = _wait_for_answer(session_id, 0)
        return DeepSeekAnswer(answer=answer, duration_seconds=time.monotonic() - started)
    finally:
        if session_id:
            with contextlib.suppress(DeepSeekBridgeError):
                _run(["session", "stop", session_id], timeout=15)


__all__ = ["DeepSeekAnswer", "DeepSeekBridgeError", "ask_deepseek"]
