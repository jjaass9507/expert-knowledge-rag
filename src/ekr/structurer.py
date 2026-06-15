"""逐字稿 → 知識卡片：組裝 prompt、呼叫 LLM、驗證 YAML、失敗重試。

provenance 欄位（id / 原始逐字稿 / 更新人 / 最後更新）由程式注入，不交給 LLM，
以縮短 prompt、消除一類幻覺、保證溯源正確。
"""

from __future__ import annotations

import json
import re
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


def _structure_prompts(transcript: str) -> tuple[str, str]:
    system = _load_prompt("structure_system.txt")
    human = _load_prompt("structure_human.txt").replace("{transcript}", transcript)
    return system, human


def _strip_fence(text: str) -> str:
    """移除模型偶爾加上的 ``` 或 ```json 圍欄。"""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _extract_json_object(text: str) -> dict | None:
    """從可能夾帶說明文字的輸出中擷取第一個 JSON 物件。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _pick_fields(data: dict) -> dict:
    """只取 LLM 應負責的欄位，忽略多餘鍵。"""
    return {k: data[k] for k in LLM_FIELDS if k in data}


def _parse_llm_fields(raw: str) -> dict:
    text = _strip_fence(raw or "")
    # yaml.safe_load 相容 JSON（JSON 為 YAML 子集）。
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    # 退而求其次：模型可能在 JSON 前後夾帶說明文字，擷取第一個 {...} 物件。
    if not isinstance(data, dict):
        data = _extract_json_object(text)

    if isinstance(data, dict):
        fields = _pick_fields(data)
        if fields:
            return fields
        # 欄位可能被包一層（例如 {"output": "{...}"} 或巢狀物件），嘗試解開後再取。
        for v in data.values():
            inner = v if isinstance(v, dict) else (
                _extract_json_object(v) if isinstance(v, str) else None
            )
            if isinstance(inner, dict):
                fields = _pick_fields(inner)
                if fields:
                    return fields

    snippet = (raw or "").strip()
    if len(snippet) > 500:
        snippet = snippet[:500] + "…（已截斷）"
    found = list(data.keys()) if isinstance(data, dict) else None
    raise ValueError(
        f"LLM 輸出缺少預期欄位（標題/內容/知識類型/信心等級）。"
        f"解析到的鍵：{found}。原始輸出：{snippet!r}"
    )


def _generate(
    llm: LLM, system: str, human: str, provenance: dict, max_retries: int
) -> KnowledgeCard:
    """共用流程：呼叫 LLM、解析六欄、注入 provenance、驗證、失敗回饋重試。"""
    current_human = human
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        raw = llm.complete(system, current_human)
        try:
            fields = _parse_llm_fields(raw)
            return KnowledgeCard(**provenance, **fields)
        except (ValidationError, ValueError, yaml.YAMLError) as e:
            last_err = e
            if attempt < max_retries:
                current_human = (
                    human
                    + f"\n\n上一次輸出無法解析或驗證失敗，錯誤：{e}\n請只輸出合法 JSON 物件，並嚴格遵守欄位與列舉值要求。"
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
    system, human = _structure_prompts(transcript)
    return _generate(llm, system, human, provenance, max_retries)


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
    system = _load_prompt("refine_system.txt")
    human = (
        _load_prompt("refine_human.txt")
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
    return _generate(llm, system, human, provenance, max_retries)
