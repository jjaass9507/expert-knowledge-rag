"""知識卡片資料模型 —— 整個系統 schema 的單一來源。

以 pydantic v2 約束欄位與列舉值；非法值會 fail loudly，正好餵給結構化的重試迴圈。
中文欄名即實際 pydantic 欄位，讓 YAML 直接以中文呈現、可讀，免 alias 映射層。
"""

from __future__ import annotations

from enum import Enum

import yaml
from pydantic import BaseModel, ConfigDict


class KnowledgeType(str, Enum):
    診斷 = "診斷"
    SOP = "SOP"
    規格 = "規格"
    經驗法則 = "經驗法則"
    其他 = "其他"


class Confidence(str, Enum):
    高 = "高"
    中 = "中"
    低 = "低"


# 設備大分類預設清單（可依現場調整）；存為字串以允許新增/留空。
EQUIPMENT_CATEGORIES = [
    "空壓機",
    "冰水主機",
    "泵浦",
    "冷卻水塔",
    "鍋爐",
    "配電/電氣",
    "管路/閥件",
    "儀控",
    "其他",
]


# LLM pass-1 結構化負責的欄位；其餘 provenance 由程式注入，重點由 pass-2 萃取。
LLM_FIELDS = ("標題", "內容", "標籤", "知識類型", "大分類", "適用範圍", "信心等級", "可回答問題")


class KnowledgeCard(BaseModel):
    """一張結構化知識卡片。"""

    model_config = ConfigDict(extra="forbid")  # 拒絕幻覺欄位

    id: str
    標題: str
    內容: str
    重點: list[str] = []
    可回答問題: list[str] = []
    標籤: list[str] = []
    知識類型: KnowledgeType
    大分類: str = ""
    適用範圍: str = ""
    信心等級: Confidence
    原始逐字稿: str
    更新人: str
    最後更新: str  # ISO 日期字串，例如 2026-06-15

    @classmethod
    def from_yaml(cls, text: str) -> "KnowledgeCard":
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("YAML 內容不是物件（mapping）")
        return cls(**data)

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
