#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_formula_index_v1.py
- 从 v4_1 方剂 JSONL 构建中文句向量索引（以便快速候选召回）
- 默认模型：BAAI/bge-small-zh-v1.5
用法：
python build_formula_index_v1.py \
  --formulas /path/方剂v4_1_完整.jsonl \
  --out /path/index_bge_small_zh_v15.npz \
  --embed-model BAAI/bge-small-zh-v1.5
"""
import argparse, json, os, sys, re, numpy as np
from pathlib import Path

def load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("请先安装 sentence-transformers： pip install -U sentence-transformers", file=sys.stderr)
        raise
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model

def l2norm(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom

def formula_text_blob(f: dict) -> str:
    parts = []
    parts.append(f.get("方名",""))
    if f.get("功效"): parts.append("功效：" + f["功效"])
    if f.get("主治症状"): parts.append("主治症状：" + "；".join(f["主治症状"]))
    if f.get("临床应用"): parts.append("临床应用：" + "；".join(f["临床应用"]))
    # patterns：病证名/别名/子项
    ptexts = []
    for pb in (f.get("patterns") or []):
        name = pb.get("name"); alias = pb.get("alias")
        seg = []
        if name: seg.append(name)
        if alias: seg.append(alias)
        subs = [sp.get("text","") for sp in (pb.get("subpoints") or []) if sp.get("text")]
        if subs: seg.append("；".join(subs))
        if seg: ptexts.append("（".join(seg)+")")
    if ptexts:
        parts.append("病证：" + "。".join(ptexts))
    # 标签
    if f.get("舌诊标签"): parts.append("舌诊：" + "、".join(f["舌诊标签"]))
    if f.get("面诊标签"): parts.append("面诊：" + "、".join(f["面诊标签"]))
    return "。".join([p for p in parts if p])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formulas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    formulas = []
    with open(args.formulas, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                formulas.append(json.loads(line))

    model = load_embedder(args.embed_model)
    texts = [formula_text_blob(f) for f in formulas]
    embs = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True)
    embs = embs.astype("float32")

    meta = {
        "id": [f.get("ID") for f in formulas],
        "name": [f.get("方名") for f in formulas],
        "source": [f.get("出处") for f in formulas],
        "功效": [f.get("功效") for f in formulas],
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, embeddings=embs, **meta)
    print(f"索引已保存：{outp} ；共 {len(formulas)} 条")

if __name__ == "__main__":
    main()
