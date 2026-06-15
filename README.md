# expert-knowledge-rag

讓現場專家以最低成本（口述或簡短文字）提供知識，系統自動結構化為 RAG 知識卡片，
經輕量審核後寫入知識庫。核心原則：**專家負責「表達」，LLM 負責「結構化」，專家只需「確認」**。

## 架構

```
入口層(文字/語音) → ASR → LLM 結構化 → 審核介面(校稿台) → 知識庫(JSONL/YAML) → 向量庫 → 檢索(RAG)
```

## 分階段

| 階段 | 內容 | 狀態 |
|---|---|---|
| Phase 1 | 文字表單 + LLM 結構化 + Flask 校稿台審核 | ✅ 已完成 |
| Phase 2 | 語音輸入 + Whisper ASR | ✅ 已完成 |
| Phase 3 | 向量化 pipeline（embedding + Qdrant） | ✅ 已完成 |
| Phase 4 | 對話式「退回重整理」 | ✅ 已完成 |

## 快速開始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 Pensieve 端點/token；離線可設 EKR_LLM=stub
python run.py                 # http://localhost:8000
```

- `/submit` 貼上逐字稿或上傳音檔 → 自動結構化為卡片草稿
- `/` 待校稿清單 → 進入校稿台逐欄校對 →「確認，加入知識庫」/「退回重整理」(可填補充說明)
- `/library` 知識卡目錄 → 瀏覽已定稿卡片
- 核准卡片寫入 `data/approved/cards.jsonl`（向量化來源）與 `data/approved/yaml/<id>.yaml`（git 溯源）

## 檢索（RAG）

啟用向量化（`.env` 設 `EKR_VECTOR=on` 並填 `EMBEDDING_*`/`QDRANT_*`）後：

```bash
python -m scripts.reindex                      # 從 cards.jsonl 回填/重建向量庫
python -m scripts.search "電流偏高但壓力正常"    # 檢索
python -m scripts.search "電流偏高" --type 診斷  # 依知識類型過濾
```

程式內呼叫：`ekr.vectorstore.retrieve(query, embedding, store, top_k, 知識類型)` 回傳
`list[Hit]`（id / score / metadata）。

## 與 8-step Agent Framework 銜接

本知識庫對應 Agent 異常評估的 **Tier 2：專家經驗/案例知識**。Agent 的 RAG 檢索節點呼叫
`retrieve(...)`，並可用 `知識類型` 做 metadata filter（如診斷場景傳 `知識類型='診斷'`）。
Tier 1（結構化規則/數值閾值）維持獨立 JSON 規則表，不經本 pipeline。

## 測試

```bash
python -m pytest -q
```

## 模組

- `src/ekr/models.py` — 知識卡片 schema（pydantic，單一來源）
- `src/ekr/llm.py` — LLM adapter（內部 Pensieve API / Stub）
- `src/ekr/asr.py` — 語音轉文字 adapter（Whisper / Stub）
- `src/ekr/structurer.py` — 逐字稿 → 卡片（驗證 + 重試）
- `src/ekr/storage.py` — SQLite 佇列 + 核准寫出
- `src/ekr/vectorstore.py` — embedding + 向量庫 + 檢索（Qdrant / InMemory）
- `src/ekr/web/` — Flask 校稿台審核介面 + 知識卡目錄
- `scripts/` — `reindex.py`（回填向量庫）、`search.py`（檢索）
