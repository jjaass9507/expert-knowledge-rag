import json

import pytest

from ekr import llm as llm_mod
from ekr.llm import OpenAILLM, PensieveLLM, available_backends, build_llm


class FakeResp:
    def __init__(self, payload, status_code=200, text="", headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_pensieve_payload_and_result_parsing(monkeypatch):
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["kwargs"] = kwargs
        result = json["variables"]  # echo 回傳供斷言
        return FakeResp(
            {"isSuccess": True, "Result": '{"標題": "x", "信心等級": "中"}'}
        )

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)

    client = PensieveLLM(
        url="http://pensieve/api", token="T", empno="E123", building="option"
    )
    out = client.complete("系統指示", "使用者輸入")

    # 回傳 Result 原始字串
    assert json.loads(out)["標題"] == "x"
    # payload 結構
    p = captured["json"]
    assert p["token"] == "T"
    assert p["empno"] == "E123"
    assert p["variables"]["building"] == "option"
    assert p["variables"]["other_system_prompt"] == "系統指示"
    assert p["variables"]["other_human_prompt"] == "使用者輸入"
    # 連線參數：停用 SSL 驗證與 proxies
    assert captured["kwargs"]["verify"] is False
    assert captured["kwargs"]["proxies"] == {"http": None, "https": None}
    assert captured["kwargs"]["timeout"] == 300


def test_pensieve_raises_on_failure(monkeypatch):
    monkeypatch.setattr(
        llm_mod.requests,
        "post",
        lambda *a, **k: FakeResp({"isSuccess": False, "msg": "boom"}),
    )
    client = PensieveLLM(url="u", token="T", empno="E")
    with pytest.raises(ValueError):
        client.complete("s", "h")


def test_openai_payload_and_parsing(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["kwargs"] = kwargs
        return FakeResp({"choices": [{"message": {"content": "結果文字"}}]})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    client = OpenAILLM(url="http://x/v1/chat/completions", model="m1", api_key="ask_123")
    out = client.complete("系統", "使用者")

    assert out == "結果文字"
    p = captured["json"]
    assert p["model"] == "m1"
    assert p["messages"] == [
        {"role": "system", "content": "系統"},
        {"role": "user", "content": "使用者"},
    ]
    assert captured["headers"]["Authorization"] == "Bearer ask_123"
    assert captured["kwargs"]["verify"] is False  # 內部自簽憑證


def test_openai_retries_on_429_then_succeeds(monkeypatch):
    OK = FakeResp({"choices": [{"message": {"content": "成功"}}]})
    RATE = FakeResp({"error": {"type": "rate_limit_error"}}, status_code=429,
                    text="rate limited", headers={"Retry-After": "20"})
    responses = [RATE, RATE, OK]
    calls = {"post": 0, "slept": []}

    def fake_post(*a, **k):
        r = responses[calls["post"]]
        calls["post"] += 1
        return r

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: calls["slept"].append(s))

    client = OpenAILLM(url="u", model="m", max_retries=5)
    assert client.complete("s", "h") == "成功"
    assert calls["post"] == 3          # 兩次 429 + 一次成功
    assert calls["slept"] == [20, 20]  # 遵守 Retry-After 標頭


def test_openai_429_exhausts_retries_then_raises(monkeypatch):
    RATE = FakeResp({"error": {}}, status_code=429, text="rate limited")
    calls = {"post": 0}

    def fake_post(*a, **k):
        calls["post"] += 1
        return RATE

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)

    client = OpenAILLM(url="u", model="m", max_retries=2)  # 無 Retry-After → 指數退避
    with pytest.raises(ValueError, match="429"):
        client.complete("s", "h")
    assert calls["post"] == 3  # 初次 + 2 次重試後仍 429 → 拋出


def test_openai_non_429_error_does_not_retry(monkeypatch):
    calls = {"post": 0}

    def fake_post(*a, **k):
        calls["post"] += 1
        return FakeResp({"error": {}}, status_code=400, text="bad request")

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    client = OpenAILLM(url="u", model="m", max_retries=5)
    with pytest.raises(ValueError, match="400"):
        client.complete("s", "h")
    assert calls["post"] == 1  # 400 立即拋出，不重試


def test_openai_logs_request_and_response(monkeypatch, caplog):
    import logging

    def fake_post(url, json=None, headers=None, **kwargs):
        return FakeResp({"choices": [{"message": {"content": "回應內容XYZ"}}]})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    client = OpenAILLM(url="u", model="m")
    with caplog.at_level(logging.DEBUG, logger="ekr.llm"):
        client.complete("你是測試助理SYS", "使用者問題HUMAN")
    text = caplog.text
    assert "你是測試助理SYS" in text  # 送出的 system 有被記錄
    assert "使用者問題HUMAN" in text  # 送出的 human 有被記錄
    assert "回應內容XYZ" in text       # 模型原始回應有被記錄


def test_available_backends_and_build(monkeypatch):
    monkeypatch.setenv("OPENAI_API_URL", "http://x")
    monkeypatch.setenv("OPENAI_MODEL", "m1")
    monkeypatch.delenv("PENSIEVE_URL", raising=False)
    ids = [b[0] for b in available_backends()]
    assert "openai" in ids and "stub" in ids
    assert isinstance(build_llm("openai"), OpenAILLM)
