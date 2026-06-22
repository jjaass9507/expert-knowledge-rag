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


def test_structure_transcripts_unwraps_object_wrapper():
    from ekr.structurer import structure_transcripts
    wrapped = ('{"cards":[' + CARD_ARRAY[1:-1] + ']}')
    cards = structure_transcripts("x", StubLLM(wrapped), "技師")
    assert len(cards) == 2
    assert cards[0].標題 == "調高源頭壓力會增加耗電"


def test_structure_transcripts_english_keys_mapped():
    from ekr.structurer import structure_transcripts
    eng = ('[{"title":"電流偏高","content":"壓力正常電流高","key_points":["檢查潤滑"],'
           '"tags":["電流"],"type":"診斷","equipment":"冰水主機","scope":"","confidence":"中"}]')
    cards = structure_transcripts("x", StubLLM(eng), "技師")
    assert len(cards) == 1
    assert cards[0].標題 == "電流偏高"
    assert cards[0].知識類型.value == "診斷"
    assert cards[0].大分類 == "冰水主機"
    assert cards[0].重點 == ["檢查潤滑"]


def test_structure_transcripts_simplified_values_mapped():
    from ekr.structurer import structure_transcripts
    simp = ('[{"标题":"测试","内容":"内容","重点":["点一"],"标签":["标"],'
            '"知识类型":"经验法则","大分类":"空压机","适用范围":"","信心等级":"高"}]')
    cards = structure_transcripts("x", StubLLM(simp), "技師")
    assert cards[0].知識類型.value == "經驗法則"  # 简体值 → 繁體
    assert cards[0].大分類 == "空壓機"


def test_structure_transcripts_minimal_keys_with_defaults():
    # 模型只給「知識點/說明」→ 映射為標題/內容, 其餘填預設, 仍成卡
    minimal = ('[{"知識點":"源頭壓力與耗電","說明":"每升1 bar耗電增約7%"},'
               '{"知識點":"新SOP","說明":"改裝感測器找瓶頸"}]')
    from ekr.structurer import structure_transcripts
    cards = structure_transcripts("x", StubLLM(minimal), "技師")
    assert len(cards) == 2
    assert cards[0].標題 == "源頭壓力與耗電"
    assert cards[0].內容 == "每升1 bar耗電增約7%"
    assert cards[0].知識類型.value == "其他"  # 缺漏 → 預設
    assert cards[0].信心等級.value == "中"


META_JSON = ('{"重點":["補的重點"],"可回答問題":["補的問題？"],"標籤":["補標籤"],'
             '"知識類型":"診斷","大分類":"泵浦","適用範圍":"P-101","信心等級":"高"}')


def test_two_stage_extract_then_structure():
    from ekr.structurer import structure_transcripts

    class TwoStage:
        def __init__(self):
            self.stages = []

        def complete(self, system, human):
            if "知識工程師" in system:  # 階段一：萃取知識單元
                self.stages.append("extract")
                return '{"知識": ["陳述A", "陳述B"]}'
            if "補齊" in system:  # 階段三：補全空白欄位
                self.stages.append("complete")
                return META_JSON
            self.stages.append("structure")  # 階段二：結構化成卡
            return CARD_ARRAY

    llm = TwoStage()
    cards = structure_transcripts("一大段口述", llm, "Ken")
    assert llm.stages[:2] == ["extract", "structure"]  # 先萃取再結構化
    assert "complete" in llm.stages  # 補全 pass 有跑
    assert len(cards) == 2
    assert cards[0].標題 == "調高源頭壓力會增加耗電"
    # 補全只填空白欄位：CARD_ARRAY 已有重點 → 不覆蓋；可回答問題原為空 → 補上。
    assert cards[0].重點 == ["壓力每升1 bar耗電增約7%"]
    assert cards[0].可回答問題 == ["補的問題？"]


def test_completion_fills_sparse_card():
    """模型在 stage-2 只給標題/內容 → 補全 pass 為其餘欄位產出初版。"""
    from ekr.structurer import structure_transcripts

    class Sparse:
        def complete(self, system, human):
            if "知識工程師" in system:
                return '{"知識": ["命題A"]}'
            if "補齊" in system:
                return META_JSON
            return '[{"標題":"只有標題","內容":"只有內容"}]'  # stage-2 稀疏輸出

    cards = structure_transcripts("一段口述", Sparse(), "Ken")
    assert len(cards) == 1
    c = cards[0]
    assert c.標題 == "只有標題" and c.內容 == "只有內容"
    assert c.重點 == ["補的重點"]
    assert c.可回答問題 == ["補的問題？"]
    assert c.標籤 == ["補標籤"]
    assert c.知識類型.value == "診斷"      # 由預設「其他」被補成推斷值
    assert c.大分類 == "泵浦"
    assert c.適用範圍 == "P-101"
    assert c.信心等級.value == "高"          # 由預設「中」被補成推斷值


def test_falls_back_to_direct_when_no_units():
    from ekr.structurer import structure_transcripts

    class NoUnits:
        def complete(self, system, human):
            if "知識工程師" in system:
                return '{"知識": []}'  # 萃取不到 → 後備直接結構化
            return CARD_ARRAY

    cards = structure_transcripts("x", NoUnits(), "K")
    assert len(cards) == 2


def test_structure_transcripts_single_object_wrapped():
    from ekr.structurer import structure_transcripts
    # 模型只回單一物件（非陣列）→ 包成一張卡
    cards = structure_transcripts("只談一件事", StubLLM(GOOD_JSON), "技師")
    assert len(cards) == 1
    assert cards[0].標題 == "電流偏高但壓力正常"


def test_structure_transcripts_topic_content_keys_mapped():
    # 重現 OpenAI 後端回傳 topic/content/source 的情形 → 仍能成卡
    from ekr.structurer import structure_transcripts
    out = ('[{"topic":"專案背景","content":"舊有做法是隨意調高源頭壓力","source":"逐字稿"},'
           '{"topic":"新SOP","content":"改裝壓力感測器找瓶頸","source":"逐字稿"}]')
    cards = structure_transcripts("一大段口述", StubLLM(out), "技師")
    assert len(cards) == 2
    assert cards[0].標題 == "專案背景"  # topic → 標題
    assert cards[0].內容 == "舊有做法是隨意調高源頭壓力"  # content → 內容
    assert cards[0].知識類型.value == "其他"  # 缺漏 → 預設
    assert cards[0].信心等級.value == "中"


def test_structure_transcripts_parses_可回答問題():
    from ekr.structurer import structure_transcripts
    out = ('[{"標題":"調高源頭壓力會增加耗電","內容":"每升1 bar耗電增約7%",'
           '"可回答問題":["調高壓力會增加多少耗電？","為什麼不該隨意調高源頭壓力？"],'
           '"知識類型":"經驗法則","信心等級":"高"}]')
    cards = structure_transcripts("x", StubLLM(out), "技師")
    assert cards[0].可回答問題 == ["調高壓力會增加多少耗電？", "為什麼不該隨意調高源頭壓力？"]


def test_human_prompt_contains_transcript():
    system, human = _structure_prompts("這是逐字稿內容")
    assert "這是逐字稿內容" in human
    assert "知識管理助理" in system  # system 帶規則


# --- qwen3 推理模型情境：輸出夾帶 <think>…</think>，且推理段含括號/大括號/冒號 ---
# （這些字元會讓舊的貪婪正則 \{.*\} / \[.*\] 抓錯範圍而解析失敗，正是欄位全空的主因）
THINK = (
    "<think>\n讓我分析：這張卡需要哪些欄位 [標題, 內容, 重點]？"
    "範例物件可能長這樣 {鍵: 值}，我先想清楚再輸出最終 JSON。\n</think>\n"
)


def test_strip_think_removes_reasoning_keeping_json():
    from ekr.structurer import _strip_fence, _strip_think
    assert _strip_think(THINK + GOOD_JSON).startswith("{")
    # 經 _strip_fence（含剝除推理 + 去圍欄）後，仍是可解析的 JSON
    assert json.loads(_strip_fence(THINK + GOOD_JSON))["標題"] == "電流偏高但壓力正常"
    # 省略開頭 <think>、只有結尾 </think> 也能處理
    assert _strip_think("一堆推理…</think>\n" + GOOD_JSON).startswith("{")


def test_extract_knowledge_units_with_think_wrapper():
    from ekr.structurer import _extract_knowledge_units
    raw = THINK + '{"知識": ["命題A", "命題B"]}'
    units = _extract_knowledge_units("逐字稿", StubLLM(raw))
    assert units == ["命題A", "命題B"]  # 成功萃取，未因推理段而走後備回 []


def test_complete_card_fills_through_think_wrapper():
    from ekr.models import Confidence, KnowledgeCard, KnowledgeType
    from ekr.structurer import complete_card
    sparse = KnowledgeCard(
        id="KB-test", 標題="只有標題", 內容="只有內容",
        知識類型=KnowledgeType.其他, 信心等級=Confidence.中,
        原始逐字稿="x", 更新人="K", 最後更新="2026-01-01",
    )
    out = complete_card(sparse, StubLLM(THINK + META_JSON))
    assert out.可回答問題 == ["補的問題？"]
    assert out.標籤 == ["補標籤"]
    assert out.知識類型.value == "診斷"


def test_structure_transcripts_skips_completion_when_rich():
    """stage-2 已產出三清單皆備的完整卡 → 不再發補全（補齊）呼叫，省呼叫、避免 429。"""
    from ekr.structurer import structure_transcripts

    rich = ('[{"標題":"調高源頭壓力會增加耗電","內容":"每升1 bar耗電增約7%",'
            '"重點":["每升1 bar耗電+7%"],"標籤":["壓力","耗電"],'
            '"可回答問題":["調高壓力會多耗多少電？"],"知識類型":"經驗法則","信心等級":"高"}]')

    class Recorder:
        def __init__(self):
            self.saw_complete = False

        def complete(self, system, human):
            if "知識工程師" in system:
                return '{"知識": ["命題A"]}'
            if "補齊" in system:
                self.saw_complete = True
                return META_JSON
            return rich

    llm = Recorder()
    cards = structure_transcripts("x", llm, "Ken")
    assert llm.saw_complete is False          # 完整卡 → 跳過補全呼叫
    assert cards[0].重點 == ["每升1 bar耗電+7%"]
    assert cards[0].可回答問題 == ["調高壓力會多耗多少電？"]


def test_structure_transcripts_end_to_end_with_think():
    """三次 LLM 呼叫皆夾帶推理段 → 剝除後仍能解析，卡片欄位填滿。"""
    from ekr.structurer import structure_transcripts

    sparse_array = '[{"標題":"調高源頭壓力會增加耗電","內容":"每升1 bar耗電增約7%。"}]'

    class ThinkingQwen:
        def complete(self, system, human):
            if "知識工程師" in system:               # 階段一
                return THINK + '{"知識": ["命題A"]}'
            if "補齊" in system:                      # 補全 pass
                return THINK + META_JSON
            return THINK + sparse_array              # 階段二（稀疏輸出）

    cards = structure_transcripts("一段口述", ThinkingQwen(), "Ken")
    assert len(cards) == 1
    c = cards[0]
    assert c.標題 == "調高源頭壓力會增加耗電"
    assert c.重點 == ["補的重點"]
    assert c.可回答問題 == ["補的問題？"]
    assert c.標籤 == ["補標籤"]
    assert c.知識類型.value == "診斷"
    assert c.大分類 == "泵浦"
    assert c.信心等級.value == "高"
