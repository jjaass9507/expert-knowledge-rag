"""LLM adapter —— 唯一刻意保留的抽象。

正式環境走內部 Pensieve API（payload 帶 token/empno、以 variables.building 路由、
回應雙層 isSuccess+Result）；測試/離線用 StubLLM。
adapter 只做 (system, human) prompt 進、文字出，不認得知識卡片；prompt 由 structurer 組裝。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol

import requests

log = logging.getLogger(__name__)

# verify=False 會發出 InsecureRequestWarning，停用以免日誌噪音。
try:
    from urllib3.exceptions import InsecureRequestWarning

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass


class LLM(Protocol):
    def complete(self, system: str, human: str) -> str: ...


class PensieveLLM:
    """內部 Pensieve API。building 為路由變數，值 'option' 走可自訂 system/human prompt 的路徑。"""

    def __init__(
        self,
        url: str,
        token: str,
        empno: str,
        building: str = "option",
        verify_ssl: bool = False,
        timeout: int = 300,
    ):
        self.url = url
        self.token = token
        self.empno = empno
        self.building = building
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    def complete(self, system: str, human: str) -> str:
        payload = {
            "token": self.token,
            "empno": self.empno,
            "variables": {
                "building": self.building,
                # option 路由以 other_system_prompt / other_human_prompt 接收自訂 prompt
                "other_system_prompt": system,
                "other_human_prompt": human,
            },
        }
        resp = requests.post(
            self.url,
            json=payload,
            verify=self.verify_ssl,
            proxies={"http": None, "https": None},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        api_response = resp.json()
        if not api_response.get("isSuccess"):
            raise ValueError(f"Pensieve API 回傳失敗：{api_response}")
        # Result 為 JSON 字串，交給 structurer 解析（yaml.safe_load 相容 JSON）。
        return api_response.get("Result", "")


class StubLLM:
    """測試/離線用：忽略輸入，依序回傳預設回應。"""

    def __init__(self, *responses: str):
        if not responses:
            raise ValueError("StubLLM 需至少一個回應")
        self._responses = list(responses)
        self._i = 0

    def complete(self, system: str, human: str) -> str:
        i = min(self._i, len(self._responses) - 1)
        self._i += 1
        return self._responses[i]


class OpenAILLM:
    """OpenAI 相容 /chat/completions API（Authorization: Bearer）。"""

    # 速率限制(429)／暫時過載(503) 才重試；其餘錯誤立即拋出。
    _RETRYABLE = (429, 503)

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        verify_ssl: bool = False,
        timeout: int = 300,
        json_mode: bool = False,
        max_retries: int = 5,
        backoff_base: float = 5.0,
        backoff_cap: float = 60.0,
    ):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap

    def _wait_seconds(self, resp, attempt: int) -> float:
        """優先遵守 Retry-After 標頭（數字秒）；否則指數退避。"""
        retry_after = (getattr(resp, "headers", None) or {}).get("Retry-After")
        try:
            if retry_after is not None:
                return min(self.backoff_cap, float(retry_after))
        except (TypeError, ValueError):
            pass  # 非數字（如 HTTP-date）則改用退避
        return min(self.backoff_cap, self.backoff_base * 2 ** attempt)

    def complete(self, system: str, human: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": human},
            ],
        }
        if self.json_mode:  # 選用；閘道需支援才開（OPENAI_JSON_MODE=true）
            payload["response_format"] = {"type": "json_object"}

        resp = None
        for attempt in range(self.max_retries + 1):
            resp = requests.post(
                self.url,
                json=payload,
                headers=headers,
                verify=self.verify_ssl,
                proxies={"http": None, "https": None},
                timeout=self.timeout,
            )
            if resp.status_code in self._RETRYABLE and attempt < self.max_retries:
                wait = self._wait_seconds(resp, attempt)
                log.warning(
                    "OpenAI API %s 速率限制，第 %d 次重試前等待 %.0fs",
                    resp.status_code, attempt + 1, wait,
                )
                time.sleep(wait)
                continue
            break

        # 官方錯誤為非 2xx + {"error":{...}}；把回傳內容帶進例外便於診斷。
        if resp.status_code >= 400:
            raise ValueError(f"OpenAI API 回傳 {resp.status_code}：{resp.text[:500]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"OpenAI API 未回傳 choices：{data}")
        content = (choices[0].get("message") or {}).get("content")
        if not content:
            # 內容過濾／長度截斷／tool 呼叫等情形 content 可能為 null
            reason = choices[0].get("finish_reason")
            raise ValueError(f"OpenAI API 回傳空內容（finish_reason={reason}）")
        return content


# --- 後端選擇 ---
def available_backends() -> list[tuple[str, str]]:
    """回傳已設定的 LLM 後端 [(id, 顯示名稱), ...]，供平台下拉選單使用。"""
    out: list[tuple[str, str]] = []
    if os.environ.get("PENSIEVE_URL"):
        out.append(("pensieve", os.environ.get("PENSIEVE_LABEL") or "Pensieve · GPT-4.1-mini"))
    if os.environ.get("OPENAI_API_URL"):
        out.append(("openai", os.environ.get("OPENAI_LABEL") or "OpenAI 相容 API"))
    out.append(("stub", "離線測試 (Stub)"))
    return out


def build_llm(backend: str) -> LLM:
    """依後端 id 建立 LLM 實例。"""
    if backend == "openai":
        return OpenAILLM(
            url=os.environ["OPENAI_API_URL"],
            model=os.environ["OPENAI_MODEL"],
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            verify_ssl=os.environ.get("OPENAI_VERIFY_SSL", "false").lower() == "true",
            timeout=int(os.environ.get("OPENAI_TIMEOUT", "300")),
            json_mode=os.environ.get("OPENAI_JSON_MODE", "false").lower() == "true",
            max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "5")),
        )
    if backend == "stub":
        return StubLLM(_STUB_JSON)
    return PensieveLLM(
        url=os.environ["PENSIEVE_URL"],
        token=os.environ["PENSIEVE_TOKEN"],
        empno=os.environ["EMPNO"],
        building=os.environ.get("PENSIEVE_BUILDING", "option"),
        verify_ssl=os.environ.get("PENSIEVE_VERIFY_SSL", "false").lower() == "true",
        timeout=int(os.environ.get("PENSIEVE_TIMEOUT", "300")),
    )


def from_env() -> LLM:
    """依環境變數建立預設 LLM；EKR_LLM 指定後端（pensieve / openai / stub）。"""
    return build_llm(os.environ.get("EKR_LLM", "pensieve").lower())


_STUB_JSON = """\
{
  "標題": "電流偏高但壓力正常的研判",
  "內容": "當系統壓力顯示正常、但運轉電流持續偏高並飄動時，研判多為壓縮機負載異常或潤滑不足所致，建議先檢查潤滑與軸承狀況。",
  "標籤": ["電流", "壓縮機", "潤滑"],
  "知識類型": "診斷",
  "適用範圍": "",
  "信心等級": "中"
}
"""
