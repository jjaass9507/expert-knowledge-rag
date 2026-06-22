"""端到端診斷：把一段逐字稿跑完整結構化流程，並印出每次 LLM 呼叫的請求/回應與最終卡片。

開啟 DEBUG logging，因此會看到每個 pass（萃取 / 結構化 / 補全）實際送出的 system+human prompt
與模型原始回應，最後逐張印出卡片所有欄位。用於確認「送了什麼、模型回了什麼、欄位是否補滿」。

    python -m scripts.trace "壓力正常但電流偏高並飄動"
    python -m scripts.trace --file 逐字稿.txt
    type 逐字稿.txt | python -m scripts.trace        # 由 stdin 讀入
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from ekr.llm import from_env  # noqa: E402
from ekr.logging_setup import setup_logging  # noqa: E402
from ekr.structurer import structure_transcripts  # noqa: E402

_FIELDS = (
    "標題", "內容", "重點", "可回答問題", "標籤",
    "知識類型", "大分類", "適用範圍", "信心等級",
)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="結構化流程診斷（顯示 LLM 請求/回應與卡片欄位）")
    parser.add_argument("transcript", nargs="?", help="逐字稿；省略則由 --file 或 stdin 讀入")
    parser.add_argument("--file", help="從檔案讀取逐字稿")
    args = parser.parse_args()

    if args.transcript:
        transcript = args.transcript
    elif args.file:
        transcript = Path(args.file).read_text(encoding="utf-8")
    else:
        transcript = sys.stdin.read()
    transcript = transcript.strip()
    if not transcript:
        parser.error("沒有逐字稿輸入（請給字串、--file 或 stdin）")

    setup_logging("DEBUG")  # 強制顯示每次 LLM 請求/回應
    llm = from_env()
    print(f"\n=== 後端：{type(llm).__name__}；逐字稿 {len(transcript)} 字 ===\n", file=sys.stderr)

    cards = structure_transcripts(transcript, llm, "診斷")

    print(f"\n=== 產出 {len(cards)} 張卡片 ===")
    for i, card in enumerate(cards, 1):
        data = {f: getattr(card, f) for f in _FIELDS}
        data["知識類型"] = card.知識類型.value
        data["信心等級"] = card.信心等級.value
        print(f"\n--- 卡片 {i} ---")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        empty = [f for f in _FIELDS if not getattr(card, f) and f != "適用範圍"]
        if empty:
            print(f"⚠ 仍為空的欄位：{empty}")


if __name__ == "__main__":
    main()
