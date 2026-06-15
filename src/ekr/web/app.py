"""Flask 審核介面 —— 「校稿台 / 編輯室」。

路由薄薄一層覆在 storage / structurer 之上：清單、提交、左右校對審核、核准/退回/重整理。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for

from ..llm import from_env
from ..models import Confidence, KnowledgeType
from ..storage import Storage
from ..structurer import structure_transcript

load_dotenv()


def create_app(storage: Storage | None = None) -> Flask:
    app = Flask(__name__)
    store = storage or Storage()

    KNOWLEDGE_TYPES = [t.value for t in KnowledgeType]
    CONFIDENCES = [c.value for c in Confidence]

    @app.route("/")
    def index():
        pending = store.list_by_status("pending")
        approved = store.list_by_status("approved")
        return render_template(
            "list.html", pending=pending, approved_count=len(approved)
        )

    @app.route("/submit", methods=["GET", "POST"])
    def submit():
        if request.method == "POST":
            transcript = request.form.get("逐字稿", "").strip()
            更新人 = request.form.get("更新人", "").strip() or "匿名"
            if not transcript:
                return render_template(
                    "submit.html", error="請輸入逐字稿或描述內容", 更新人=更新人
                )
            try:
                card = structure_transcript(transcript, from_env(), 更新人)
            except Exception as e:  # noqa: BLE001 — 結構化失敗如實回報給審核者
                return render_template(
                    "submit.html",
                    error=f"結構化失敗：{e}",
                    逐字稿=transcript,
                    更新人=更新人,
                )
            store.insert_pending(card)
            return redirect(url_for("review", card_id=card.id))
        return render_template("submit.html")

    @app.route("/review/<card_id>")
    def review(card_id):
        card = store.get(card_id)
        if card is None:
            return "找不到此卡片", 404
        return render_template(
            "review.html",
            card=card,
            knowledge_types=KNOWLEDGE_TYPES,
            confidences=CONFIDENCES,
        )

    @app.route("/review/<card_id>/approve", methods=["POST"])
    def approve(card_id):
        _save_edits(store, card_id, request.form)
        store.approve(card_id)
        return redirect(url_for("index"))

    @app.route("/review/<card_id>/reject", methods=["POST"])
    def reject(card_id):
        store.reject(card_id)
        return redirect(url_for("index"))

    @app.route("/review/<card_id>/restructure", methods=["POST"])
    def restructure(card_id):
        card = store.get(card_id)
        if card is None:
            return "找不到此卡片", 404
        # 允許審核者補充逐字稿後重新結構化
        transcript = request.form.get("原始逐字稿", "").strip() or card.原始逐字稿
        try:
            new = structure_transcript(transcript, from_env(), card.更新人)
        except Exception as e:  # noqa: BLE001
            return render_template(
                "review.html",
                card=card,
                knowledge_types=KNOWLEDGE_TYPES,
                confidences=CONFIDENCES,
                error=f"重新整理失敗：{e}",
            )
        store.update_fields(
            card_id,
            標題=new.標題,
            內容=new.內容,
            標籤=new.標籤,
            知識類型=new.知識類型.value,
            適用範圍=new.適用範圍,
            信心等級=new.信心等級.value,
        )
        return redirect(url_for("review", card_id=card_id))

    return app


def _save_edits(store: Storage, card_id: str, form) -> None:
    標籤 = [t.strip() for t in form.get("標籤", "").split(",") if t.strip()]
    store.update_fields(
        card_id,
        標題=form.get("標題", "").strip(),
        內容=form.get("內容", "").strip(),
        標籤=標籤,
        知識類型=form.get("知識類型", ""),
        適用範圍=form.get("適用範圍", "").strip(),
        信心等級=form.get("信心等級", ""),
    )


if __name__ == "__main__":
    create_app().run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
