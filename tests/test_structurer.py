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


def test_human_prompt_contains_transcript():
    system, human = _structure_prompts("這是逐字稿內容")
    assert "這是逐字稿內容" in human
    assert "知識管理助理" in system  # system 帶規則
