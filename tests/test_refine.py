from ekr.llm import StubLLM
from ekr.models import KnowledgeCard
from ekr.structurer import refine_card

CARD = KnowledgeCard(
    id="KB-ref01",
    標題="電流偏高",
    內容="壓力正常但電流偏高。",
    標籤=["電流"],
    知識類型="診斷",
    適用範圍="",
    信心等級="中",
    原始逐字稿="壓力正常但電流飄高，一定是壓縮機",
    更新人="王技師",
    最後更新="2026-06-15",
)

REFINED_YAML = """\
標題: 電流偏高但壓力正常的研判
內容: 壓力正常但電流偏高並飄動，研判壓縮機負載異常。
標籤: [電流, 壓縮機]
知識類型: 診斷
適用範圍: RTHD 冰水主機
信心等級: 高
"""


def test_refine_preserves_provenance_and_applies_changes():
    new = refine_card(CARD, "信心改高，補上型號 RTHD", StubLLM(REFINED_YAML))
    # provenance 保留
    assert new.id == "KB-ref01"
    assert new.原始逐字稿 == CARD.原始逐字稿
    assert new.更新人 == "王技師"
    # 內容依補充說明更新
    assert new.信心等級.value == "高"
    assert "RTHD" in new.適用範圍


def test_refine_prompt_carries_card_and_feedback():
    captured = {}

    class Capture:
        def complete(self, prompt):
            captured["prompt"] = prompt
            return REFINED_YAML

    refine_card(CARD, "請補上型號", Capture())
    assert "壓力正常但電流偏高" in captured["prompt"]  # 目前卡片
    assert "請補上型號" in captured["prompt"]  # 補充說明
    assert CARD.原始逐字稿 in captured["prompt"]  # 原始逐字稿
