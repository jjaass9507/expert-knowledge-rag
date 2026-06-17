"""儲存層：SQLite 待審佇列 + 核准後寫出 JSONL/YAML。

- 待審佇列與審核狀態存於 SQLite（status: pending / approved / rejected）。
- 核准時重新以 KnowledgeCard 驗證（防審核者手改破壞 schema），再 append 一行到
  approved/cards.jsonl（Phase 3 向量化的來源），並寫一份 yaml/<id>.yaml 供 git 溯源。
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import KnowledgeCard

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = _ROOT / "data" / "cards.db"
DEFAULT_APPROVED_DIR = _ROOT / "data" / "approved"

STATUSES = ("pending", "approved", "rejected")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    標題 TEXT NOT NULL,
    內容 TEXT NOT NULL,
    重點 TEXT NOT NULL DEFAULT '[]',  -- JSON array
    可回答問題 TEXT NOT NULL DEFAULT '[]',  -- JSON array
    標籤 TEXT NOT NULL,            -- JSON array
    知識類型 TEXT NOT NULL,
    大分類 TEXT NOT NULL DEFAULT '',
    適用範圍 TEXT NOT NULL DEFAULT '',
    信心等級 TEXT NOT NULL,
    原始逐字稿 TEXT NOT NULL,
    更新人 TEXT NOT NULL,
    最後更新 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

# 既有資料庫的欄位補丁（欄名 → 欄定義）。
_MIGRATIONS = {
    "重點": "TEXT NOT NULL DEFAULT '[]'",
    "可回答問題": "TEXT NOT NULL DEFAULT '[]'",
    "大分類": "TEXT NOT NULL DEFAULT ''",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB,
        approved_dir: Path | str = DEFAULT_APPROVED_DIR,
        on_approve=None,
        on_delete=None,
    ):
        # on_approve(card)：核准/編輯定稿卡片後觸發（向量索引）；None 則略過。
        # on_delete(card_id)：刪除定稿卡片後觸發（移除索引）；None 則略過。
        self.on_approve = on_approve
        self.on_delete = on_delete
        self.db_path = Path(db_path)
        self.approved_dir = Path(approved_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(cards)")}
            for col, ddl in _MIGRATIONS.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE cards ADD COLUMN {col} {ddl}")

    @contextlib.contextmanager
    def _connect(self):
        """每次操作各開一條連線，避免 SQLite 連線跨執行緒（Flask 多執行緒）的問題。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        # 不再持有長壽連線，保留以相容呼叫端。
        pass

    # --- 設定（key-value）---
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # --- 寫入 ---
    def insert_pending(self, card: KnowledgeCard) -> None:
        ts = _now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO cards (id, 標題, 內容, 重點, 可回答問題, 標籤, 知識類型, 大分類,
                   適用範圍, 信心等級, 原始逐字稿, 更新人, 最後更新,
                   status, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    card.id,
                    card.標題,
                    card.內容,
                    json.dumps(card.重點, ensure_ascii=False),
                    json.dumps(card.可回答問題, ensure_ascii=False),
                    json.dumps(card.標籤, ensure_ascii=False),
                    card.知識類型.value,
                    card.大分類,
                    card.適用範圍,
                    card.信心等級.value,
                    card.原始逐字稿,
                    card.更新人,
                    card.最後更新,
                    "pending",
                    ts,
                    ts,
                ),
            )

    def update_fields(self, card_id: str, **fields) -> None:
        """更新審核者編輯過的欄位（標題/內容/標籤/知識類型/適用範圍/信心等級）。"""
        allowed = {"標題", "內容", "重點", "可回答問題", "標籤", "知識類型", "大分類", "適用範圍", "信心等級"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k in ("標籤", "重點", "可回答問題") and isinstance(v, list):
                v = json.dumps(v, ensure_ascii=False)
            sets.append(f"{k}=?")
            vals.append(v)
        if not sets:
            return
        sets.append("updated_at=?")
        vals.append(_now())
        vals.append(card_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE cards SET {', '.join(sets)} WHERE id=?", vals)

    def set_status(self, card_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"未知狀態：{status}")
        with self._connect() as conn:
            conn.execute(
                "UPDATE cards SET status=?, updated_at=? WHERE id=?",
                (status, _now(), card_id),
            )

    # --- 讀取 ---
    def get(self, card_id: str) -> KnowledgeCard | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cards WHERE id=?", (card_id,)
            ).fetchone()
        return _row_to_card(row) if row else None

    def list_by_status(self, status: str) -> list[KnowledgeCard]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cards WHERE status=? ORDER BY created_at DESC", (status,)
            ).fetchall()
        return [_row_to_card(r) for r in rows]

    # --- 審核動作 ---
    def approve(self, card_id: str) -> KnowledgeCard:
        card = self.get(card_id)
        if card is None:
            raise KeyError(card_id)
        # 重新驗證（防手改破壞 schema）
        card = KnowledgeCard(**card.model_dump())
        self.set_status(card_id, "approved")
        self._write_card_yaml(card)
        self._rewrite_jsonl()
        if self.on_approve is not None:
            self.on_approve(card)
        return card

    def reject(self, card_id: str) -> None:
        if self.get(card_id) is None:
            raise KeyError(card_id)
        self.set_status(card_id, "rejected")

    def update_card(self, card_id: str, **fields) -> KnowledgeCard:
        """編輯卡片欄位；若為定稿卡片，同步更新 YAML/JSONL 並重新索引。"""
        if self.get(card_id) is None:
            raise KeyError(card_id)
        self.update_fields(card_id, **fields)
        card = self.get(card_id)
        # 驗證編輯後仍合法
        card = KnowledgeCard(**card.model_dump())
        with self._connect() as conn:
            status = conn.execute(
                "SELECT status FROM cards WHERE id=?", (card_id,)
            ).fetchone()["status"]
        if status == "approved":
            self._write_card_yaml(card)
            self._rewrite_jsonl()
            if self.on_approve is not None:
                self.on_approve(card)
        return card

    def delete(self, card_id: str) -> None:
        """刪除卡片；若為定稿卡片，同步移除 YAML/JSONL 與向量索引。"""
        if self.get(card_id) is None:
            raise KeyError(card_id)
        with self._connect() as conn:
            status = conn.execute(
                "SELECT status FROM cards WHERE id=?", (card_id,)
            ).fetchone()["status"]
            conn.execute("DELETE FROM cards WHERE id=?", (card_id,))
        if status == "approved":
            yaml_file = self.approved_dir / "yaml" / f"{card_id}.yaml"
            if yaml_file.exists():
                yaml_file.unlink()
            self._rewrite_jsonl()
            if self.on_delete is not None:
                self.on_delete(card_id)

    def _write_card_yaml(self, card: KnowledgeCard) -> None:
        (self.approved_dir / "yaml").mkdir(parents=True, exist_ok=True)
        (self.approved_dir / "yaml" / f"{card.id}.yaml").write_text(
            card.to_yaml(), encoding="utf-8"
        )

    def _rewrite_jsonl(self) -> None:
        """由 SQLite 的定稿卡片重新產生 cards.jsonl（編輯/刪除後保持一致）。"""
        self.approved_dir.mkdir(parents=True, exist_ok=True)
        approved = self.list_by_status("approved")
        with (self.approved_dir / "cards.jsonl").open("w", encoding="utf-8") as f:
            for card in approved:
                f.write(
                    json.dumps(card.model_dump(mode="json"), ensure_ascii=False) + "\n"
                )


def _row_to_card(row: sqlite3.Row) -> KnowledgeCard:
    return KnowledgeCard(
        id=row["id"],
        標題=row["標題"],
        內容=row["內容"],
        重點=json.loads(row["重點"]) if row["重點"] else [],
        可回答問題=json.loads(row["可回答問題"]) if row["可回答問題"] else [],
        標籤=json.loads(row["標籤"]),
        知識類型=row["知識類型"],
        大分類=row["大分類"] if row["大分類"] else "",
        適用範圍=row["適用範圍"],
        信心等級=row["信心等級"],
        原始逐字稿=row["原始逐字稿"],
        更新人=row["更新人"],
        最後更新=row["最後更新"],
    )
