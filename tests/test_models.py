import pytest
from pydantic import ValidationError

from ekr.models import KnowledgeCard

GOOD = {
    "id": "KB-0001",
    "標題": "電流偏高研判",
    "內容": "壓力正常但電流偏高，研判壓縮機負載異常。",
    "標籤": ["電流", "壓縮機"],
    "知識類型": "診斷",
    "適用範圍": "",
    "信心等級": "中",
    "原始逐字稿": "就是壓力都正常啦，但是電流會飄高...",
    "更新人": "王技師",
    "最後更新": "2026-06-15",
}


def test_valid_card_roundtrips():
    card = KnowledgeCard(**GOOD)
    text = card.to_yaml()
    again = KnowledgeCard.from_yaml(text)
    assert again.標題 == "電流偏高研判"
    assert again.知識類型.value == "診斷"


def test_to_yaml_keeps_chinese():
    text = KnowledgeCard(**GOOD).to_yaml()
    assert "電流偏高研判" in text
    assert "\\u" not in text  # allow_unicode 生效，非 \uXXXX


def test_missing_field_raises():
    bad = {k: v for k, v in GOOD.items() if k != "標題"}
    with pytest.raises(ValidationError):
        KnowledgeCard(**bad)


def test_bad_enum_raises():
    bad = {**GOOD, "知識類型": "亂寫"}
    with pytest.raises(ValidationError):
        KnowledgeCard(**bad)


def test_extra_field_rejected():
    bad = {**GOOD, "亂入欄位": "x"}
    with pytest.raises(ValidationError):
        KnowledgeCard(**bad)
