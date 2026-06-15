"""逐字稿 → 知識卡片：組裝 prompt、呼叫 LLM、驗證 YAML、失敗重試。

provenance 欄位（id / 原始逐字稿 / 更新人 / 最後更新）由程式注入，不交給 LLM，
以縮短 prompt、消除一類幻覺、保證溯源正確。
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import yaml
from pydantic import ValidationError

from .llm import LLM
from .models import LLM_FIELDS, KnowledgeCard

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _assemble_prompt(transcript: str) -> str:
    return _load_prompt("structure_card.txt").replace("{transcript}", transcript)


def _strip_fence(text: str) -> str:
    """移除模型偶爾加上的 ``` 或 ```yaml 圍欄。"""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _parse_llm_fields(raw: str) -> dict:
    data = yaml.safe_load(_strip_fence(raw))
    if not isinstance(data, dict):
        raise ValueError("LLM 輸出不是 YAML 物件")
    # 只取 LLM 應負責的欄位，忽略多餘鍵。
    return {k: data[k] for k in LLM_FIELDS if k in data}


def _generate(llm: LLM, base_prompt: str, provenance: dict, max_retries: int) -> KnowledgeCard:
    """共用流程：呼叫 LLM、解析六欄、注入 provenance、驗證、失敗回饋重試。"""
    prompt = base_prompt
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        raw = llm.complete(prompt)
        try:
            fields = _parse_llm_fields(raw)
            return KnowledgeCard(**provenance, **fields)
        except (ValidationError, ValueError, yaml.YAMLError) as e:
            last_err = e
            if attempt < max_retries:
                prompt = (
                    base_prompt
                    + f"\n\n上一次輸出無法解析或驗證失敗，錯誤：{e}\n請只輸出合法 YAML，並嚴格遵守欄位與列舉值要求。"
                )
    raise ValueError(f"結構化失敗（已重試 {max_retries} 次）：{last_err}")


def structure_transcript(
    transcript: str,
    llm: LLM,
    更新人: str,
    now: str | None = None,
    max_retries: int = 1,
) -> KnowledgeCard:
    """將逐字稿結構化為一張已驗證的 KnowledgeCard。"""
    provenance = {
        "id": "KB-" + uuid.uuid4().hex[:8],
        "原始逐字稿": transcript,
        "更新人": 更新人,
        "最後更新": now or date.today().isoformat(),
    }
    return _generate(llm, _assemble_prompt(transcript), provenance, max_retries)


def refine_card(
    card: KnowledgeCard,
    feedback: str,
    llm: LLM,
    now: str | None = None,
    max_retries: int = 1,
) -> KnowledgeCard:
    """依審核者補充說明，對話式重整理既有卡片；保留 id/原始逐字稿/更新人。"""
    d = card.model_dump(mode="json")
    current = "\n".join(
        f"{k}: {d[k]}"
        for k in ("標題", "內容", "標籤", "知識類型", "適用範圍", "信心等級")
    )
    base_prompt = (
        _load_prompt("refine_card.txt")
        .replace("{transcript}", card.原始逐字稿)
        .replace("{current}", current)
        .replace("{feedback}", feedback)
    )
    provenance = {
        "id": card.id,
        "原始逐字稿": card.原始逐字稿,
        "更新人": card.更新人,
        "最後更新": now or date.today().isoformat(),
    }
    return _generate(llm, base_prompt, provenance, max_retries)
