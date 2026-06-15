"""檢索知識庫（示範 RAG 檢索，亦為 Agent Tier 2 檢索節點的對應入口）。

需先設定 .env 的 EMBEDDING_* 與 QDRANT_*。

    python -m scripts.search "電流偏高但壓力正常"
    python -m scripts.search "電流偏高" --type 診斷 --top-k 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from ekr.vectorstore import build_embedding_store, retrieve  # noqa: E402


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="檢索知識卡片")
    parser.add_argument("query", help="查詢字串")
    parser.add_argument("--type", dest="知識類型", default=None, help="依知識類型過濾")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embedding, store = build_embedding_store()
    hits = retrieve(args.query, embedding, store, top_k=args.top_k, 知識類型=args.知識類型)
    if not hits:
        print("（無結果）")
        return
    for i, h in enumerate(hits, 1):
        md = h.metadata
        print(f"{i}. [{h.score:.3f}] {h.id} · {md.get('知識類型')} · 信心{md.get('信心等級')}")
        print(f"   {md.get('標題')}")
        print(f"   {md.get('內容')}\n")


if __name__ == "__main__":
    main()
