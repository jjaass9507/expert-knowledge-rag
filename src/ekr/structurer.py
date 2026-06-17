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

# 欄位鍵別名 → 正規中文鍵（容忍簡體中文與英文鍵，以及常見的精簡命名）。
_KEY_ALIASES = {
    "标题": "標題", "title": "標題", "知識點": "標題", "知识点": "標題",
    "主題": "標題", "主题": "標題", "topic": "標題", "subject": "標題", "heading": "標題",
    "内容": "內容", "content": "內容", "說明": "內容", "说明": "內容",
    "摘要": "內容", "描述": "內容", "summary": "內容", "description": "內容",
    "重点": "重點", "key_points": "重點", "keypoints": "重點",
    "points": "重點", "highlights": "重點",
    "可回答的問題": "可回答問題", "可回答问题": "可回答問題", "問題": "可回答問題",
    "问题": "可回答問題", "questions": "可回答問題", "faq": "可回答問題",
    "标签": "標籤", "tags": "標籤", "labels": "標籤", "keywords": "標籤",
    "知识类型": "知識類型", "type": "知識類型", "knowledge_type": "知識類型",
    "大分类": "大分類", "equipment": "大分類", "equipment_category": "大分類",
    "category": "大分類", "domain": "大分類",
    "适用范围": "適用範圍", "scope": "適用範圍", "applicable_scope": "適用範圍",
    "信心等级": "信心等級", "confidence": "信心等級", "confidence_level": "信心等級",
}

# 列舉/分類值別名 → 正規繁體值（容忍簡體）。
_VALUE_ALIASES = {
    "诊断": "診斷", "规格": "規格", "经验法则": "經驗法則",
    "空压机": "空壓機", "冰水主机": "冰水主機", "冷却水塔": "冷卻水塔",
    "锅炉": "鍋爐", "配电/电气": "配電/電氣", "管路/阀件": "管路/閥件", "仪控": "儀控",
}


def _normalize_keys(d: dict) -> dict:
    """把卡片物件的鍵正規化為繁體中文鍵（容忍簡體/英文）。"""
    out = {}
    for k, v in d.items():
        key = k.strip() if isinstance(k, str) else k
        out[_KEY_ALIASES.get(key, key)] = v
    return out


def _has_card_fields(d: dict) -> bool:
    return bool(set(_normalize_keys(d)) & set(LLM_FIELDS))

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


def _extract_json_array(text: str) -> list | None:
    """從可能夾帶說明文字的輸出中擷取第一個 JSON 陣列。"""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, list) else None


def _pick_fields(data: dict) -> dict:
    """單卡路徑：只取 LLM 應負責的欄位（不做鍵正規化，維持嚴格缺欄位偵測）；清單欄位轉字串陣列。"""
    out = {k: data[k] for k in LLM_FIELDS if k in data}
    for list_field in ("重點", "可回答問題"):
        if list_field in data:  # 重點不在 LLM_FIELDS，需直接從來源 dict 取
            out[list_field] = _as_str_list(data[list_field])
    return out


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
    """把列舉/受限欄位收斂到合法值（容忍簡體）；自創類別/等級落到安全預設，交審核者修正。"""
    t = fields.get("知識類型")
    if isinstance(t, str):
        t = _VALUE_ALIASES.get(t.strip(), t.strip())
        fields["知識類型"] = t if t in _VALID_TYPES else "其他"
    c = fields.get("信心等級")
    if isinstance(c, str) and c.strip() not in _VALID_CONF:
        fields["信心等級"] = "中"
    # 大分類：容忍簡體；非預設清單內則清空，交審核者用下拉選擇。
    g = fields.get("大分類")
    if isinstance(g, str):
        g = _VALUE_ALIASES.get(g.strip(), g.strip())
        fields["大分類"] = g if (not g or g in EQUIPMENT_CATEGORIES) else ""
    return fields


def _fill_defaults(fields: dict) -> dict:
    """模型若只給標題與內容，填入安全預設讓卡片能成立，交審核者於校稿台補全。"""
    fields.setdefault("知識類型", "其他")
    fields.setdefault("信心等級", "中")
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


def complete_card(card: KnowledgeCard, llm: LLM, max_retries: int = 1) -> KnowledgeCard:
    """補全 pass：依標題+內容，用 LLM 為「空白／預設」的結構化欄位產一版初稿。

    只填目前缺漏的欄位，保留 stage-2 已給的有效值；任一步失敗則優雅降級回原卡。
    """
    try:
        sys = _load_prompt("complete_card_system.txt")
        hum = (
            _load_prompt("complete_card_human.txt")
            .replace("{標題}", card.標題)
            .replace("{內容}", card.內容)
        )
        fields = _normalize_enums(_pick_card_fields(_call_json(llm, sys, hum, max_retries)))
    except (ValueError, yaml.YAMLError):
        return card

    updates: dict = {}
    for f in ("重點", "可回答問題", "標籤"):
        if not getattr(card, f) and fields.get(f):
            updates[f] = _as_str_list(fields[f])
    for f in ("大分類", "適用範圍"):
        v = fields.get(f)
        if not getattr(card, f) and isinstance(v, str) and v.strip():
            updates[f] = v.strip()
    # 列舉欄位：僅在目前為預設值（其他／中）時，以推斷出的非預設值取代。
    if card.知識類型.value == "其他" and fields.get("知識類型") not in (None, "其他"):
        updates["知識類型"] = fields["知識類型"]
    if card.信心等級.value == "中" and fields.get("信心等級") not in (None, "中"):
        updates["信心等級"] = fields["信心等級"]
    if not updates:
        return card
    # 以 KnowledgeCard 重建以重新驗證（model_copy 不會把字串值轉回列舉）。
    data = card.model_dump()
    data.update(updates)
    return KnowledgeCard(**data)


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


def _pick_card_fields(d: dict) -> dict:
    """多卡路徑：先正規化鍵，再取出結構化欄位（清單欄位轉為字串陣列）。"""
    return _pick_fields(_normalize_keys(d))


def _dict_to_items(d: dict) -> list[dict]:
    """把一個 dict 轉成卡片物件清單：本身是卡片→[d]；被包一層（如 {"cards":[...]}）→取出內層陣列。"""
    if _has_card_fields(d):
        return [d]
    nested = next(
        (v for v in d.values()
         if isinstance(v, list) and any(isinstance(x, dict) for x in v)),
        None,
    )
    if nested is not None:
        return [x for x in nested if isinstance(x, dict)]
    return [d]


def _parse_card_array(raw: str) -> list[dict]:
    """把 LLM 輸出解析為卡片物件陣列。容忍：陣列、單一物件、被物件包一層、夾帶說明文字。"""
    text = _strip_fence(raw or "")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    if isinstance(data, list):
        items = [d for d in data if isinstance(d, dict)]
    elif isinstance(data, dict):
        items = _dict_to_items(data)
    else:
        arr = _extract_json_array(text)
        if isinstance(arr, list):
            items = [d for d in arr if isinstance(d, dict)]
        else:
            obj = _extract_json_object(text)
            items = _dict_to_items(obj) if isinstance(obj, dict) else []
    if not items:
        raise ValueError(f"LLM 未輸出卡片陣列。原始輸出：{_snippet(raw)}")
    return items


def _snippet(raw: str, limit: int = 600) -> str:
    s = (raw or "").strip()
    return f"{s[:limit]}…（已截斷）" if len(s) > limit else s


def _extract_knowledge_units(transcript: str, llm: LLM, max_retries: int = 1) -> list[str]:
    """階段一：判斷「何為知識」並萃取為可行動的知識陳述清單；失敗則回空清單（走後備）。"""
    try:
        sys = _load_prompt("knowledge_units_system.txt")
        hum = _load_prompt("knowledge_units_human.txt").replace("{transcript}", transcript)
        data = _call_json(llm, sys, hum, max_retries)
        return _as_str_list(
            data.get("知識") or data.get("知识") or data.get("knowledge") or data.get("units")
        )
    except (ValueError, yaml.YAMLError):
        return []


def _cards_from_array_call(
    system: str,
    base_human: str,
    llm: LLM,
    transcript: str,
    更新人: str,
    最後更新: str,
    max_retries: int,
) -> list[KnowledgeCard]:
    """呼叫 LLM 取得卡片陣列並建成 KnowledgeCard；含正規化、補預設、缺欄位重試與診斷。"""
    human = base_human
    last_err: Exception | None = None
    last_keys = None
    last_raw = ""
    for attempt in range(max_retries + 1):
        raw = llm.complete(system, human)
        last_raw = raw
        try:
            items = _parse_card_array(raw)
        except ValueError as e:
            last_err = e
            human = base_human + "\n\n請只輸出 JSON 陣列，每個知識點一個物件。"
            continue

        cards: list[KnowledgeCard] = []
        for item in items:
            fields = _fill_defaults(_normalize_enums(_pick_card_fields(item)))
            if not fields.get("標題") or not fields.get("內容"):
                continue
            try:
                cards.append(
                    KnowledgeCard(
                        id="KB-" + uuid.uuid4().hex[:8],
                        原始逐字稿=transcript,
                        更新人=更新人,
                        最後更新=最後更新,
                        **fields,
                    )
                )
            except ValidationError as e:
                last_err = e
        if cards:
            return cards
        last_keys = list(items[0].keys()) if items else None
        human = (
            base_human
            + f"\n\n上次輸出的物件鍵為 {last_keys}，缺少必要欄位。"
            + "請改用這些「繁體中文」鍵：標題、內容、重點、標籤、知識類型、大分類、適用範圍、信心等級，並輸出 JSON 陣列。"
        )

    if last_keys is not None:
        raise ValueError(
            f"結構化失敗：卡片物件缺少預期欄位。物件實際鍵：{last_keys}。原始輸出：{_snippet(last_raw)}"
        )
    raise ValueError(f"結構化失敗：{last_err}。原始輸出：{_snippet(last_raw)}")


def structure_transcripts(
    transcript: str,
    llm: LLM,
    更新人: str,
    now: str | None = None,
    max_retries: int = 1,
) -> list[KnowledgeCard]:
    """兩階段把一段口述解構為多張知識卡：

    階段一：萃取「可重複應用的知識」陳述（判斷何為知識、濾掉非知識）。
    階段二：把每條知識陳述結構化為一張可存入 RAG 的自足知識卡。
    若階段一萃取不到知識，後備為直接從逐字稿結構化。
    """
    最後更新 = now or date.today().isoformat()
    units = _extract_knowledge_units(transcript, llm, max_retries)
    if units:
        system = _load_prompt("structure_units_system.txt")
        base_human = _load_prompt("structure_units_human.txt").replace(
            "{units}", "\n".join(f"{i}. {u}" for i, u in enumerate(units, 1))
        )
    else:
        # 後備：直接從逐字稿產卡
        system = _load_prompt("structure_multi_system.txt")
        base_human = _load_prompt("structure_multi_human.txt").replace(
            "{transcript}", transcript
        )
    cards = _cards_from_array_call(
        system, base_human, llm, transcript, 更新人, 最後更新, max_retries
    )
    # 補全 pass：為每張卡的空白／預設欄位用 LLM 產一版初稿（重點/可回答問題/標籤/類型/分類/範圍/信心）。
    return [complete_card(c, llm, max_retries) for c in cards]


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
