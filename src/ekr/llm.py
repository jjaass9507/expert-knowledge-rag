"""LLM adapter —— 唯一刻意保留的抽象。

由「無 GPU、走 API」這個真實限制驅動：正式環境用 OpenAI 相容的 /chat/completions
端點（本地推論伺服器如 Ollama/vLLM/LM Studio 或雲端皆適用），測試用 StubLLM。
adapter 只做 text-in / text-out，不認得知識卡片，prompt 由 structurer 組裝。
"""

from __future__ import annotations

import os
from typing import Protocol

import requests


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class ApiLLM:
    """OpenAI 相容 chat/completions API。端點/模型/金鑰全走參數或環境變數。"""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.2,
        timeout: int = 120,
    ):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }
        resp = requests.post(
            self.url, json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class StubLLM:
    """測試/離線用：依序回傳預設回應。"""

    def __init__(self, *responses: str):
        if not responses:
            raise ValueError("StubLLM 需至少一個回應")
        self._responses = list(responses)
        self._i = 0

    def complete(self, prompt: str) -> str:
        i = min(self._i, len(self._responses) - 1)
        self._i += 1
        return self._responses[i]


def from_env() -> LLM:
    """依環境變數建立 LLM；EKR_LLM=stub 時回傳離線假回應。"""
    if os.environ.get("EKR_LLM", "api").lower() == "stub":
        return StubLLM(_STUB_YAML)
    return ApiLLM(
        url=os.environ["LLM_API_URL"],
        model=os.environ["LLM_MODEL"],
        api_key=os.environ.get("LLM_API_KEY", ""),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
    )


_STUB_YAML = """\
標題: 電流偏高但壓力正常的研判
內容: 當系統壓力顯示正常、但運轉電流持續偏高並飄動時，研判多為壓縮機負載異常或潤滑不足所致，建議先檢查潤滑與軸承狀況。
標籤: [電流, 壓縮機, 潤滑]
知識類型: 診斷
適用範圍: ""
信心等級: 中
"""
