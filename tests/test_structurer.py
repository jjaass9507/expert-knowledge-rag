import json

import pytest

from ekr.llm import StubLLM
from ekr.structurer import _structure_prompts, structure_transcript

GOOD_JSON = """\
{
  "標題": "電流偏高但壓力正常",
  "內容": "壓力正常但電流偏高並飄動，研判壓縮機負載異常。",
  "標籤": ["電流", "壓縮機"],
  "知識類型": "診斷",
  "適用範圍": "",
  "信心等級": "中"
}
"""

FENCED_JSON = "```json\n" + GOOD_JSON + "```"

BAD_YAML = "標題: 缺很多欄位\n知識類型: 亂寫\n"


def test_good_yaml_produces_card_with_injected_provenance():
    card = structure_transcript("壓力正常但電流飄高", StubLLM(GOOD_JSON), "王技師")
    assert card.標題 == "電流偏高但壓力正常"
    assert card.id.startswith("KB-")
    assert card.更新人 == "王技師"
    assert card.原始逐字稿 == "壓力正常但電流飄高"
    assert card.最後更新  # 由程式填入


def test_strips_code_fence():
    card = structure_transcript("x", StubLLM(FENCED_JSON), "技師")
    assert card.知識類型.value == "診斷"


def test_retry_recovers_after_bad_then_good():
    llm = StubLLM(BAD_YAML, GOOD_JSON)
    card = structure_transcript("x", llm, "技師", max_retries=1)
    assert card.標題 == "電流偏高但壓力正常"


def test_always_bad_raises():
    llm = StubLLM(BAD_YAML)
    with pytest.raises(ValueError):
        structure_transcript("x", llm, "技師", max_retries=1)


def test_extracts_json_when_wrapped_in_prose():
    wrapped = "好的，以下是整理結果：\n" + GOOD_JSON + "\n希望有幫助。"
    card = structure_transcript("x", StubLLM(wrapped), "技師")
    assert card.標題 == "電流偏高但壓力正常"


def test_unwraps_one_level_wrapper():
    wrapped = '{"output": ' + json.dumps(GOOD_JSON) + "}"
    card = structure_transcript("x", StubLLM(wrapped), "技師")
    assert card.標題 == "電流偏高但壓力正常"


def test_unparseable_output_surfaces_raw_in_error():
    llm = StubLLM("抱歉，我無法處理這個請求。")
    with pytest.raises(ValueError, match="原始輸出"):
        structure_transcript("x", llm, "技師", max_retries=0)


def test_missing_fields_surfaces_keys_and_raw():
    # 解析成 dict 但鍵不符（例如英文鍵）
    llm = StubLLM('{"title": "x", "content": "y"}')
    with pytest.raises(ValueError, match="解析到的鍵"):
        structure_transcript("x", llm, "技師", max_retries=0)


def test_大分類_outside_preset_cleared():
    bad = """{
      "標題": "x", "內容": "y", "標籤": [], "知識類型": "其他",
      "大分類": "不存在的設備", "適用範圍": "", "信心等級": "中"
    }"""
    card = structure_transcript("x", StubLLM(bad), "技師")
    assert card.大分類 == ""   # 非預設清單 → 清空交審核者選


def test_invalid_enum_values_coerced_to_safe_defaults():
    bad = """{
      "標題": "電流偏高",
      "內容": "壓力正常但電流偏高。",
      "標籤": ["電流"],
      "知識類型": "技術知識",
      "適用範圍": "",
      "信心等級": "很高"
    }"""
    card = structure_transcript("x", StubLLM(bad), "技師")
    assert card.知識類型.value == "其他"   # 自創類別 → 其他
    assert card.信心等級.value == "中"     # 非法等級 → 中


CARD_ARRAY = """[
  {"標題":"調高源頭壓力會增加耗電","內容":"源頭壓力每調高1 bar，空壓機總耗電約增加7%。","重點":["壓力每升1 bar耗電增約7%"],"標籤":["壓力","耗電"],"知識類型":"經驗法則","大分類":"空壓機","適用範圍":"","信心等級":"高"},
  {"標題":"推行不調高源頭壓力的新SOP","內容":"改在主幹管與機台末端裝壓力感測器找瓶頸，不再隨意調高源頭壓力。","重點":["裝壓力感測器找瓶頸"],"標籤":["SOP","感測器"],"知識類型":"SOP","大分類":"空壓機","適用範圍":"","信心等級":"中"}
]"""


def test_structure_transcripts_array_makes_multiple_cards():
    from ekr.structurer import structure_transcripts
    cards = structure_transcripts("一大段口述", StubLLM(CARD_ARRAY), "王技師")
    assert len(cards) == 2
    assert cards[0].標題 == "調高源頭壓力會增加耗電"
    assert cards[0].重點 == ["壓力每升1 bar耗電增約7%"]
    assert cards[1].知識類型.value == "SOP"
    assert all(c.原始逐字稿 == "一大段口述" for c in cards)  # 皆保留來源
    assert cards[0].id != cards[1].id


def test_structure_transcripts_single_object_wrapped():
    from ekr.structurer import structure_transcripts
    # 模型只回單一物件（非陣列）→ 包成一張卡
    cards = structure_transcripts("只談一件事", StubLLM(GOOD_JSON), "技師")
    assert len(cards) == 1
    assert cards[0].標題 == "電流偏高但壓力正常"


def test_human_prompt_contains_transcript():
    system, human = _structure_prompts("這是逐字稿內容")
    assert "這是逐字稿內容" in human
    assert "知識管理助理" in system  # system 帶規則
