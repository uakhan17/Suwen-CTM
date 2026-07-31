# -*- coding: utf-8 -*-
# bm25_utils.py
# 轻量 BM25 支持（中文友好）：基于 rank_bm25 + 可选 jieba；提供构建索引与查询函数。
from typing import List, Dict, Any, Tuple
import re

try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    raise RuntimeError("缺少 rank_bm25，请先安装：pip install rank-bm25") from e

try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

def _ngram_chars(s: str, n: int = 2) -> List[str]:
    # 将连续的中文字符串切成 n-gram，英文/数字保留整体
    out: List[str] = []
    for seg in re.findall(r"[\\u4e00-\\u9fff]+|[A-Za-z0-9_]+", s):
        if re.search(r"[\\u4e00-\\u9fff]", seg):
            out.extend([seg[i:i+n] for i in range(max(len(seg)-n+1, 1))])
        else:
            out.append(seg.lower())
    return out

def tokenize(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if _HAS_JIEBA:
        # 使用精确模式，避免过度切分；并兼容中英文混排
        return [t.strip().lower() for t in jieba.lcut(text, cut_all=False, HMM=False) if t.strip()]
    # 无 jieba 时，回退到 2-gram 字符切分
    return _ngram_chars(text, n=2)

def _join_fields(formula: Dict[str, Any]) -> str:
    # 将一个方剂的关键信息串接成索引文本
    parts: List[str] = []
    # 常用字段
    for k in ["方名","功效","出处","用法","剂型","剂量","临床应用","注意","主治病证_raw"]:
        v = formula.get(k)
        if isinstance(v, list):
            parts.append("，".join([str(x) for x in v if x]))
        elif v:
            parts.append(str(v))
    # 组成药味
    comp = formula.get("composition") or []
    herbs = [it.get("药味","") for it in comp if it.get("药味")]
    if herbs:
        parts.append("、".join(herbs))
    # 标签
    for k in ["舌诊标签","面诊标签"]:
        v = formula.get(k)
        if isinstance(v, list):
            parts.append("，".join([str(x) for x in v if x]))
        elif v:
            parts.append(str(v))
    # patterns 展开
    pats = formula.get("patterns") or []
    for p in pats:
        parts.extend([str(p.get("name") or ""), str(p.get("alias") or "")])
        for sp in p.get("subpoints") or []:
            parts.append(str(sp.get("text") or ""))
    return "。".join([x for x in parts if x])

def build_bm25_index(formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
    corpus_texts: List[str] = [_join_fields(f) for f in formulas]
    corpus_tokens: List[List[str]] = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(corpus_tokens)
    return {
        "bm25": bm25,
        "corpus_tokens": corpus_tokens,
        "texts": corpus_texts,
        "size": len(corpus_texts),
    }

def patient_to_query_text(patient: Dict[str, Any]) -> str:
    parts: List[str] = []
    ft = patient.get("free_text_dialogue")
    if ft: parts.append(str(ft))
    # 结构化
    smc = patient.get("structured_main_complaint") or {}
    for k in ["key_symptoms","tongue","face"]:
        v = smc.get(k) or []
        if isinstance(v, list) and v:
            parts.append("，".join([str(x) for x in v if x]))
    # 扁平字段
    for k in ["symptoms","tongue_tags","face_tags","tcm_syndrome","western_diagnoses"]:
        v = patient.get(k) or []
        if isinstance(v, list) and v:
            parts.append("，".join([str(x) for x in v if x]))
    return "。".join([x for x in parts if x])

def bm25_topn(index: Dict[str, Any], query_text: str, topn: int = 30) -> Tuple[List[int], List[float]]:
    tokens = tokenize(query_text)
    scores = index["bm25"].get_scores(tokens)
    # 取前 topn 下标与分数
    import numpy as np
    idx = np.argsort(scores)[::-1][: min(topn, len(scores))]
    vals = [float(scores[i]) for i in idx]
    return list(idx), vals

def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values); hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
