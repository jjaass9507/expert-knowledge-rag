"""從 data/approved/cards.jsonl 回填/重建向量庫。

用於：初次啟用向量化前已累積的卡片、或更換 embedding 模型後重建。
需先設定 .env 的 EMBEDDING_* 與 QDRANT_*。

    python -m scripts.reindex
    python -m scripts.reindex --jsonl data/approved/cards.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from ekr.storage import DEFAULT_APPROVED_DIR  # noqa: E402
from ekr.vectorstore import build_embedding_store, reindex_jsonl  # noqa: E402


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="回填知識卡片到向量庫")
    parser.add_argument(
        "--jsonl",
        default=str(DEFAULT_APPROVED_DIR / "cards.jsonl"),
        help="cards.jsonl 路徑",
    )
    args = parser.parse_args()

    embedding, store = build_embedding_store()
    n = reindex_jsonl(args.jsonl, embedding, store)
    print(f"已索引 {n} 張知識卡片到向量庫。")


if __name__ == "__main__":
    main()
