# -*- coding: utf-8 -*-
# recommend_bm25_wrapper.py
# 将 BM25 召回融入现有 recommend_v2 浞程：先“BM25 + 向量”联合召回，再调用原函数重排；最后用 BM25 分数做一次融合重排。
from typing import Dict, Any, List, Tuple
import numpy as np

from bm25_utils import build_bm25_index, patient_to_query_text, bm25_topn, minmax_norm

# 引入你已有的 v2 模块
from recommend_v2 import recommend_for_patient

# 缓存，避免重复构建
_BM25_CACHE = {"key": None, "index": None}

def _ensure_bm25(formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _BM25_CACHE
    key = (id(formulas), len(formulas))
    if _BM25_CACHE["key"] != key or _BM25_CACHE["index"] is None:
        _BM25_CACHE = {"key": key, "index": build_bm25_index(formulas)}
    return _BM25_CACHE["index"]

def _subset(emb_mat: np.ndarray, meta: Dict[str, Any], idxs: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    # emb_mat: [N, D] -> [M, D]
    sub_emb = emb_mat[idxs, :]
    sub_meta: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        try:
            if isinstance(v, list) and len(v) == emb_mat.shape[0]:
                sub_meta[k] = [v[i] for i in idxs]
            elif hasattr(v, "shape") and getattr(v, "shape", None) and v.shape[0] == emb_mat.shape[0]:
                sub_meta[k] = v[idxs]
            else:
                sub_meta[k] = v
        except Exception:
            sub_meta[k] = v
    return sub_emb, sub_meta

def recommend_for_patient_with_bm25(
    patient: Dict[str, Any],
    formulas: List[Dict[str, Any]],
    emb_mat, meta, model,
    topk: int = 5,
    recall: int = 30,            # dense 召回候选数（保持与原版兼容）
    recall_bm25: int = 30,       # 新增：BM25 召回候选数
    bm25_weight: float = 0.3,    # 新增：融合权重（0~1）
):
    """
    返回结构与 recommend_v2.recommend_for_patient 一致；只是增加 why['bm25'] 与 why['fused_score']。
    """
    # 1) BM25 召回
    bm25_idx = _ensure_bm25(formulas)
    q = patient_to_query_text(patient)
    bm25_ids, bm25_scores = bm25_topn(bm25_idx, q, topn=recall_bm25)
    bm25_norm = minmax_norm(bm25_scores)
    bm25_dict = {i: s for i, s in zip(bm25_ids, bm25_norm)}

    # 2) Dense 召回（取 recall 数量交给原版做重排）
    # 候选集合 = BM25 前 recall_bm25；若不足，则退化为全库
    cand = set(bm25_ids)
    if len(cand) < recall:
        cand = set(range(len(formulas)))
    cand = sorted(list(cand))

    # 3) 在“候选子集”上调用原版的 recommend_for_patient（这样 Dense 召回只在子集中发生）
    sub_formulas = [formulas[i] for i in cand]
    sub_emb, sub_meta = _subset(np.asarray(emb_mat), meta, cand)
    # 注意：把 recall 设置为子集大小，确保原函数能拿到足够候选进行重排
    res = recommend_for_patient(
        patient, sub_formulas, sub_emb, sub_meta, model,
        topk=min(topk, len(sub_formulas)), recall=len(sub_formulas)
    )

    # 4) 融合 BM25 分数：对原始 score 做 min-max 归一化，与 bm25_norm 按权重融合；并重新排序
    if not res:
        return res
    orig_scores = [float(x.get("score", 0.0)) for x in res]
    s_norm = minmax_norm(orig_scores)
    fused = []
    for item, s in zip(res, s_norm):
        # 在全库中定位该方的下标（先用 ID 匹配，退化为方名匹配）
        idx_in_full = -1
        id_val = item.get("ID")
        if id_val is not None:
            for i in cand:
                if formulas[i].get("ID") == id_val:
                    idx_in_full = i
                    break
        if idx_in_full < 0:
            name = item.get("方名")
            for i in cand:
                if formulas[i].get("方名") == name:
                    idx_in_full = i
                    break
        bm25_s = bm25_dict.get(idx_in_full, 0.0)
        fused_score = (1.0 - bm25_weight) * s + bm25_weight * bm25_s
        # 写回 why
        why = item.get("why", {}) or {}
        why["bm25"] = bm25_s
        why["score_before_bm25"] = item.get("score", 0.0)
        why["fused_score"] = fused_score
        item["why"] = why
        item["score"] = fused_score
        fused.append(item)
    fused.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return fused[:topk]
