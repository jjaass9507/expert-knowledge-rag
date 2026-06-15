"""語音轉文字 adapter —— 與 LLM adapter 對稱的 stub 模式。

正式環境用本地 faster-whisper（中文 large-v3 / medium）；測試用 StubASR。
faster-whisper 為延遲匯入，未安裝時不影響純文字流程與整個 app 啟動。
"""

from __future__ import annotations

import os
from typing import Protocol


class ASR(Protocol):
    def transcribe(self, audio_path: str) -> str: ...


class WhisperASR:
    """本地 faster-whisper。可選用 initial_prompt 注入術語/型號清單做 prompt bias。"""

    def __init__(
        self,
        model_size: str = "large-v3",
        language: str = "zh",
        initial_prompt: str | None = None,
        device: str = "auto",
    ):
        self.model_size = model_size
        self.language = language
        self.initial_prompt = initial_prompt
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # 延遲匯入

            self._model = WhisperModel(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        model = self._ensure_model()
        segments, _ = model.transcribe(
            audio_path,
            language=self.language,
            initial_prompt=self.initial_prompt,
        )
        return "".join(seg.text for seg in segments).strip()


class StubASR:
    """測試/離線用：回傳預設逐字稿。"""

    def __init__(self, text: str = "壓力都正常啦，但是電流會飄高，我猜是壓縮機的問題"):
        self._text = text

    def transcribe(self, audio_path: str) -> str:
        return self._text


def from_env() -> ASR:
    """EKR_ASR=stub 時回傳離線假回應；否則建立 WhisperASR。"""
    if os.environ.get("EKR_ASR", "whisper").lower() == "stub":
        return StubASR()
    return WhisperASR(
        model_size=os.environ.get("WHISPER_MODEL", "large-v3"),
        language=os.environ.get("WHISPER_LANG", "zh"),
        initial_prompt=os.environ.get("WHISPER_PROMPT") or None,
    )
