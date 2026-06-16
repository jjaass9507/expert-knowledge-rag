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
from .models import (
    EQUIPMENT_CATEGORIES,
    LLM_FIELDS,
    Confidence,
    KnowledgeCard,
    KnowledgeType,
)

_VALID_TYPES = {t.value for t in KnowledgeType}
_VALID_CONF = {c.value for c in Confidence}

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


def _normalize_enums(fields: dict) -> dict:
    """把列舉/受限欄位收斂到合法值：模型若自創類別/等級，落到安全預設，交審核者修正。"""
    t = fields.get("知識類型")
    if isinstance(t, str) and t.strip() not in _VALID_TYPES:
        fields["知識類型"] = "其他"
    c = fields.get("信心等級")
    if isinstance(c, str) and c.strip() not in _VALID_CONF:
        fields["信心等級"] = "中"
    # 大分類非預設清單內則清空，交審核者用下拉選擇。
    g = fields.get("大分類")
    if isinstance(g, str) and g.strip() and g.strip() not in EQUIPMENT_CATEGORIES:
        fields["大分類"] = ""
    return fields


def _call_json(llm: LLM, system: str, human: str, max_retries: int = 1) -> dict:
    """呼叫 LLM 並解析為 JSON 物件（含容錯與重試）；供萃取/濃縮等附加 pass 使用。"""
    current = human
    for attempt in range(max_retries + 1):
        raw = llm.complete(system, current)
        text = _strip_fence(raw or "")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = None
        if not isinstance(data, dict):
            data = _extract_json_object(text)
        if isinstance(data, dict):
            return data
        current = human + "\n\n請只輸出合法 JSON 物件。"
    raise ValueError("附加 pass 輸出無法解析為 JSON 物件")


def _as_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def enrich_card(card: KnowledgeCard, llm: LLM, max_retries: int = 1) -> KnowledgeCard:
    """附加萃取 pass：抽出「重點」條列並濃縮「內容」。任一步失敗則優雅降級，不阻斷卡片。"""
    重點: list[str] = []
    try:
        sys = _load_prompt("extract_system.txt")
        hum = (
            _load_prompt("extract_human.txt")
            .replace("{標題}", card.標題)
            .replace("{知識類型}", card.知識類型.value)
            .replace("{內容}", card.內容)
        )
        重點 = _as_str_list(_call_json(llm, sys, hum, max_retries).get("重點"))
    except (ValueError, yaml.YAMLError, KeyError):
        重點 = []

    內容 = card.內容
    if 重點:
        try:
            sys = _load_prompt("condense_system.txt")
            hum = (
                _load_prompt("condense_human.txt")
                .replace("{重點}", "\n".join(f"- {p}" for p in 重點))
                .replace("{內容}", card.內容)
            )
            condensed = _call_json(llm, sys, hum, max_retries).get("內容")
            if isinstance(condensed, str) and condensed.strip():
                內容 = condensed.strip()
        except (ValueError, yaml.YAMLError, KeyError):
            內容 = card.內容

    return card.model_copy(update={"重點": 重點, "內容": 內容})


def _generate(
    llm: LLM, system: str, human: str, provenance: dict, max_retries: int
) -> KnowledgeCard:
    """共用流程：呼叫 LLM、解析六欄、注入 provenance、驗證、失敗回饋重試。"""
    current_human = human
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        raw = llm.complete(system, current_human)
        try:
            fields = _normalize_enums(_parse_llm_fields(raw))
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
    """將單一逐字稿（片段）結構化為一張已驗證的 KnowledgeCard。"""
    provenance = {
        "id": "KB-" + uuid.uuid4().hex[:8],
        "原始逐字稿": transcript,
        "更新人": 更新人,
        "最後更新": now or date.today().isoformat(),
    }
    system, human = _structure_prompts(transcript)
    card = _generate(llm, system, human, provenance, max_retries)
    # 附加萃取：抽重點 + 濃縮內容（失敗則保留原內容、重點留空）。
    return enrich_card(card, llm, max_retries)


def split_transcript(transcript: str, llm: LLM, max_retries: int = 1) -> list[str]:
    """把一段口述拆成多個各自獨立的知識點片段；失敗或單一知識點則回傳 [transcript]。"""
    try:
        sys = _load_prompt("split_system.txt")
        hum = _load_prompt("split_human.txt").replace("{transcript}", transcript)
        data = _call_json(llm, sys, hum, max_retries)
        segs = data.get("段落") or data.get("知識點") or []
        segs = [s.strip() for s in segs if isinstance(s, str) and s.strip()]
        return segs or [transcript]
    except (ValueError, yaml.YAMLError):
        return [transcript]


def structure_transcripts(
    transcript: str,
    llm: LLM,
    更新人: str,
    now: str | None = None,
    max_retries: int = 1,
) -> list[KnowledgeCard]:
    """解構一段口述為多張知識卡：先拆分知識點，再各自結構化。

    每個片段獨立結構化；個別片段失敗則略過，全部失敗才拋出。
    """
    segments = split_transcript(transcript, llm, max_retries)
    cards: list[KnowledgeCard] = []
    last_err: Exception | None = None
    for seg in segments:
        try:
            cards.append(
                structure_transcript(seg, llm, 更新人, now=now, max_retries=max_retries)
            )
        except (ValidationError, ValueError, yaml.YAMLError) as e:
            last_err = e
    if not cards:
        raise ValueError(f"結構化失敗：{last_err}")
    return cards


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
        for k in ("標題", "內容", "標籤", "知識類型", "大分類", "適用範圍", "信心等級")
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
    refined = _generate(llm, system, human, provenance, max_retries)
    return enrich_card(refined, llm, max_retries)
