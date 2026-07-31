# 🌿 Prescription Recommendation System

This module implements a hybrid CTM (Chinese Traditional Medicine) prescription recommendation engine. It combines sparse lexical search (**BM25**), dense vector retrieval (**BGE Embeddings**), multi-dimensional re-ranking, and CTM herbal safety constraint checking.

---

## 🔄 Pipeline Architecture & Data Flow

The engine orchestrates three core modules (`bm25_utils.py`, `recommend_v2.py`, and `recommend_bm25_wrapper.py`) to process patient profiles and produce safe, interpretable formula recommendations.

```text
  ┌─────────────────────────────────────────────────────────────┐
  │ Patient Profile (Dialogue, TCM Syndrome, Tongue/Face Tags)   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 1. Query Synthesis & Hybrid Recall                      │
    │  ├─ Lexical Recall (bm25_utils.py): BM25Okapi + Jieba   │
    │  └─ Dense Recall (recommend_v2.py): BGE Sentence Embed │
    └────────────────────────────┬────────────────────────────┘
                                 │ Candidate Subset
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 2. Multi-Dimensional Re-Ranking (recommend_v2.py)       │
    │  ├─ Rule Logic: K-of-N subpoint pattern matching        │
    │  ├─ TCM & Western Alignment: Cosine sim on syndromes    │
    │  └─ Tag Matching: Direct score on inspection features   │
    └────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 3. Safety Guard Interception (recommend_v2.py)          │
    │  └─ Filter Eighteenth Incompatibilities & Nineteen      │
    │     Counteractions (十八反 / 十九畏)                      │
    └────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 4. Score Fusion & Output (recommend_bm25_wrapper.py)    │
    │  ├─ Weighted min-max fusion: (1-w)*Dense + w*BM25       │
    │  └─ Top-K Output + Explainability Payload ("why")       │
    └─────────────────────────────────────────────────────────┘
```

---

## 📄 Module Responsibilities

### 1. `bm25_utils.py` (Lexical Indexing & Retrieval)
* **Text Preparation:** Concatenates CTM formula metadata (effects, composition, indications, inspection tags) into indexable documents.
* **Tokenization:** Performs tokenization using `jieba` (with a 2-gram character split fallback).
* **Sparse Scoring:** Computes BM25Okapi relevance scores between synthesized patient queries and the formula corpus, followed by min-max score normalization.

### 2. `recommend_v2.py` (Dense Retrieval, Fine Re-Ranking & Safety Guard)
* **Dense Vector Retrieval:** Uses `BAAI/bge-small-zh-v1.5` sentence embeddings to retrieve formula candidates based on high-level semantic context.
* **Multi-Dimensional Re-Ranking:** Scores candidates across 5 criteria:
  * **Pattern Subpoints:** Evaluates rule logic (`K-of-N`, `ALL`, `ANY`) against patient symptom expressions.
  * **TCM Syndrome Alignment:** Calculates cosine similarity between patient syndrome text and formula indications.
  * **Western Diagnosis Matching:** Combines term-frequency matching with embedding similarity for clinical applications (`临床应用`).
  * **Inspection Tag Alignment:** Scores exact/partial matches for tongue and face visual diagnostic tags.
* **Safety Interception:** Scans candidate formula compositions (`composition`) against classic herbal contraindications—**Eighteen Incompatibilities (十八反)** and **Nineteen Counteractions (十九畏)**. If a violation is detected, a heavy penalty (`score - 100`) disqualifies the candidate.

### 3. `recommend_bm25_wrapper.py` (Hybrid Fusion Pipeline)
* **Hybrid Candidate Pool:** Merges top lexical hits from BM25 with dense vector candidates.
* **Score Fusion:** Min-max normalizes both dense and sparse score arrays, applying a weighted linear fusion:
  $$\text{fused\_score} = (1 - w) \cdot \text{score}_{\text{dense}} + w \cdot \text{score}_{\text{bm25}}$$
* **Explainability Payload:** Constructs a detailed `why` dictionary for each output prescription, providing complete transparency into exact pattern hits, tag match sub-scores, and safety check flags.

---

## 🖥️ Interactive Gradio Demo

An interactive **Gradio GUI interface** has been built to showcase this recommendation pipeline in real-time. It allows users to input patient dialogue, select tongue/face inspection indicators, dynamically adjust the BM25 vs. Dense fusion weights ($w$), and visually inspect the explainable reasoning payload (`why`) alongside top-ranked prescription recommendations.