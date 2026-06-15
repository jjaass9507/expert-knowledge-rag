# expert-knowledge-rag

讓現場專家以最低成本（口述或簡短文字）提供知識，系統自動結構化為 RAG 知識卡片，
經輕量審核後寫入知識庫。核心原則：**專家負責「表達」，LLM 負責「結構化」，專家只需「確認」**。

## 架構

```
入口層(文字/語音) → ASR(P2) → LLM 結構化 → 審核介面(校稿台) → 知識庫(JSONL/YAML) → 向量庫(P3)
```

## 分階段

| 階段 | 內容 | 狀態 |
|---|---|---|
| Phase 1 | 文字表單 + LLM 結構化 + Flask 校稿台審核 | ✅ 已完成 |
| Phase 2 | 語音輸入 + Whisper ASR | 規劃中 |
| Phase 3 | 向量化 pipeline（embedding + Qdrant/Chroma） | 規劃中 |
| Phase 4 | 對話式「退回重整理」 | 規劃中 |

## 快速開始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 LLM API 端點；離線可設 EKR_LLM=stub
python run.py                 # http://localhost:8000
```

- `/submit` 貼上逐字稿 → 自動結構化為卡片草稿
- `/` 待校稿清單 → 進入校稿台逐欄校對 → 「確認，加入知識庫」
- 核准卡片寫入 `data/approved/cards.jsonl`（向量化來源）與 `data/approved/yaml/<id>.yaml`（git 溯源）

## 測試

```bash
python -m pytest -q
```

## 模組

- `src/ekr/models.py` — 知識卡片 schema（pydantic，單一來源）
- `src/ekr/llm.py` — LLM adapter（API / Stub）
- `src/ekr/structurer.py` — 逐字稿 → 卡片（驗證 + 重試）
- `src/ekr/storage.py` — SQLite 佇列 + 核准寫出
- `src/ekr/web/` — Flask 校稿台審核介面
