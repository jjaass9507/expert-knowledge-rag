import pytest

from ekr.llm import StubLLM
from ekr.structurer import _assemble_prompt, structure_transcript

GOOD_YAML = """\
標題: 電流偏高但壓力正常
內容: 壓力正常但電流偏高並飄動，研判壓縮機負載異常。
標籤: [電流, 壓縮機]
知識類型: 診斷
適用範圍: ""
信心等級: 中
"""

FENCED_YAML = "```yaml\n" + GOOD_YAML + "```"

BAD_YAML = "標題: 缺很多欄位\n知識類型: 亂寫\n"


def test_good_yaml_produces_card_with_injected_provenance():
    card = structure_transcript("壓力正常但電流飄高", StubLLM(GOOD_YAML), "王技師")
    assert card.標題 == "電流偏高但壓力正常"
    assert card.id.startswith("KB-")
    assert card.更新人 == "王技師"
    assert card.原始逐字稿 == "壓力正常但電流飄高"
    assert card.最後更新  # 由程式填入


def test_strips_code_fence():
    card = structure_transcript("x", StubLLM(FENCED_YAML), "技師")
    assert card.知識類型.value == "診斷"


def test_retry_recovers_after_bad_then_good():
    llm = StubLLM(BAD_YAML, GOOD_YAML)
    card = structure_transcript("x", llm, "技師", max_retries=1)
    assert card.標題 == "電流偏高但壓力正常"


def test_always_bad_raises():
    llm = StubLLM(BAD_YAML)
    with pytest.raises(ValueError):
        structure_transcript("x", llm, "技師", max_retries=1)


def test_prompt_contains_transcript():
    prompt = _assemble_prompt("這是逐字稿內容")
    assert "這是逐字稿內容" in prompt
