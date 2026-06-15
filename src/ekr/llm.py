"""LLM adapter —— 唯一刻意保留的抽象。

正式環境走內部 Pensieve API（payload 帶 token/empno、以 variables.building 路由、
回應雙層 isSuccess+Result）；測試/離線用 StubLLM。
adapter 只做 (system, human) prompt 進、文字出，不認得知識卡片；prompt 由 structurer 組裝。
"""

from __future__ import annotations

import os
from typing import Protocol

import requests

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
                "system_prompt": system,
                "human_prompt": human,
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


def from_env() -> LLM:
    """依環境變數建立 LLM；EKR_LLM=stub 時回傳離線假回應。"""
    if os.environ.get("EKR_LLM", "pensieve").lower() == "stub":
        return StubLLM(_STUB_JSON)
    return PensieveLLM(
        url=os.environ["PENSIEVE_URL"],
        token=os.environ["PENSIEVE_TOKEN"],
        empno=os.environ["EMPNO"],
        building=os.environ.get("PENSIEVE_BUILDING", "option"),
        verify_ssl=os.environ.get("PENSIEVE_VERIFY_SSL", "false").lower() == "true",
        timeout=int(os.environ.get("PENSIEVE_TIMEOUT", "300")),
    )


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
