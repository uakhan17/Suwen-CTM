#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_v2.py
多层次检索 + 打分：
- 句向量召回（公式级向量）
- 字符串/规则匹配（病证 K-of-N / ANY / ALL + 标签 + 症状 + 临床应用）
- 句向量精排（病人证型 ↔ 方剂病证、病人西医 ↔ 方剂临床应用、自由文本 ↔ 子项文本）
- 安全拦截：十八反 / 十九畏

输出的每条推荐，包含：方名、composition(成分/用量)、出处、剂量、剂型、用法、功效 + 可解释字段 why。

用法：
python recommend_v2.py \
  --formulas /path/方剂v4_1_完整.jsonl \
  --patients /path/synthetic_patients_v2.jsonl \
  --index /path/index_bge_small_zh_v15.npz \
  --embed-model BAAI/bge-small-zh-v1.5 \
  --topk 5 --recall 30
"""
import argparse, json, re, math, sys, numpy as np
from pathlib import Path

# ====== 配置：十八反 / 十九畏（示例；建议替换为医生团队提供的数据） ======
EIGHTEEN_FAN = {
    "甘草": ["甘遂","大戟","芫花","海藻"],
    "乌头": ["半夏","瓜蒌","贝母","白蔹","白芨"],
    "藜芦": ["人参","沙参","丹参","玄参","细辛","芍药","苦参"]
}
NINETEEN_WEI = {
    "硫磺": ["朴硝"], "水银": ["砒霜"], "狼毒": ["密陀僧"], "巴豆": ["牵牛"],
    "牙硝": ["三棱"], "丁香": ["郁金"], "肉桂": ["赤石脂"], "人参": ["藜芦"]
}

def load_formulas(p):
    items=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: items.append(json.loads(line))
    return items

def load_patients(p):
    items=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: items.append(json.loads(line))
    return items

def load_index(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]
    meta = {k: data[k].tolist() for k in data.files if k != "embeddings"}
    return emb, meta

def load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("请先安装 sentence-transformers： pip install -U sentence-transformers", file=sys.stderr)
        raise
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model

def normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v/denom

def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def patient_text_blob(p: dict) -> str:
    segs=[]
    if p.get("free_text_dialogue"): segs.append(p["free_text_dialogue"])
    smc = p.get("structured_main_complaint") or {}
    if smc.get("key_symptoms"): segs.append("主诉要点：" + "、".join(smc["key_symptoms"]))
    if p.get("tcm_syndrome"): segs.append("中医证型：" + "、".join(p["tcm_syndrome"]))
    if p.get("western_diagnoses"): segs.append("西医诊断：" + "、".join(p["western_diagnoses"]))
    if p.get("tongue_tags"): segs.append("舌诊：" + "、".join(p["tongue_tags"]))
    if p.get("face_tags"): segs.append("面诊：" + "、".join(p["face_tags"]))
    return "。".join(segs)

def extract_keywords(text):
    return [t for t in re.split(r"[，,；;。.\n、：:\s]+", text or "") if t]

def match_score_terms(terms_a, terms_b, w_exact=1.0, w_sub=0.5):
    if not terms_a or not terms_b: return 0.0
    sa=set(terms_a); sb=set(terms_b)
    inter=len(sa & sb)
    sub=0
    for a in sa:
        if any((a in b and a!=b) or (b in a and a!=b) for b in sb):
            sub+=1
    return inter*w_exact + sub*w_sub

def conflicts_in_formula(composition):
    herbs=[(h.get("药味") or "").strip() for h in (composition or []) if h.get("药味")]
    herbs_set=set(herbs)
    fan_pairs=[]; wei_pairs=[]
    for a, bs in EIGHTEEN_FAN.items():
        if a in herbs_set:
            for b in bs:
                if b in herbs_set: fan_pairs.append((a,b))
    for a, bs in NINETEEN_WEI.items():
        if a in herbs_set:
            for b in bs:
                if b in herbs_set: wei_pairs.append((a,b))
    return fan_pairs, wei_pairs

def rule_satisfied(rule, matches, total):
    if total==0: return False, 0.0
    typ=(rule or {}).get("type","k_of_n")
    if typ=="any": return (matches>=1), matches/total
    if typ=="all": return (matches==total), matches/total
    k=(rule or {}).get("k", math.ceil(total/2))
    return (matches>=k), matches/total

def subpoint_match(sub_text, blob):
    if not sub_text: return False
    qs = re.findall(r"“([^”]+)”", sub_text)
    keys = qs if qs else extract_keywords(sub_text)
    return any(k in blob for k in keys)

def embed_texts(model, texts, batch_size=64):
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return embs.astype("float32")

def formula_pattern_texts(f):
    outs=[]
    for pb in (f.get("patterns") or []):
        parts=[]
        if pb.get("name"): parts.append(pb["name"])
        if pb.get("alias"): parts.append(pb["alias"])
        subs=[sp.get("text","") for sp in (pb.get("subpoints") or []) if sp.get("text")]
        if subs: parts.append("；".join(subs))
        if parts: outs.append("。".join(parts))
    return outs

def formula_western_terms(f):
    return f.get("临床应用") or []

def ensure_index(formulas, index_path, model_name):
    # 若没给索引，或索引不存在，则临时构建（内置与 build_formula_index_v1 一致逻辑）
    if index_path and Path(index_path).exists():
        emb, meta = load_index(index_path)
        return emb, meta
    # 临时构建
    model = load_embedder(model_name)
    texts = []
    for f in formulas:
        parts=[f.get("方名","")]
        if f.get("功效"): parts.append("功效："+f["功效"])
        if f.get("临床应用"): parts.append("临床应用："+"；".join(f["临床应用"]))
        for t in formula_pattern_texts(f):
            parts.append("病证："+t)
        if f.get("舌诊标签"): parts.append("舌诊："+"、".join(f["舌诊标签"]))
        if f.get("面诊标签"): parts.append("面诊："+"、".join(f["面诊标签"]))
        texts.append("。".join([p for p in parts if p]))
    emb = embed_texts(model, texts)
    meta = {"id":[f.get("ID") for f in formulas], "name":[f.get("方名") for f in formulas]}
    return emb, meta

def recommend_for_patient(p, formulas, emb_mat, meta, model, topk=5, recall=30):
    # 1) 句向量召回（患者整体文本 vs 方剂整体向量）
    blob = patient_text_blob(p)
    q = embed_texts(model, [blob])[0:1]  # (1, d)
    sims = cosine(q, emb_mat)[0]         # (n,)
    idx_top = sims.argsort()[::-1][:recall]

    # 2) 对候选做精排：规则/字符串 + 句向量细粒度
    results=[]
    for idx in idx_top:
        f = formulas[idx]
        score = 0.0
        why = {"pattern_hits":[], "tcm_embed":0.0, "western":0.0, "lex":0.0, "tags":0.0, "text_embed":0.0, "safety":[]}
        # 2.1 规则：病证子项匹配
        pats = f.get("patterns") or []
        for pb in pats:
            subs = pb.get("subpoints") or []
            m = sum(1 for sp in subs if subpoint_match(sp.get("text",""), blob))
            ok, frac = rule_satisfied(pb.get("rule") or {}, m, len(subs))
            score += m*0.8 + (2.0 if ok else 0.0) + frac*0.3
            why["pattern_hits"].append({"code": pb.get("code"), "matches": m, "total": len(subs), "ok": ok})

        # 2.2 证型向量：patient.tcm_syndrome vs 方剂病证文本
        tcm_pat = p.get("tcm_syndrome") or []
        if tcm_pat:
            form_pat_texts = formula_pattern_texts(f) or ["；".join(tcm_pat)]
            a = embed_texts(model, ["；".join(tcm_pat)])
            b = embed_texts(model, form_pat_texts)
            sim = float(cosine(a, b).max())  # 取最大匹配
            score += sim * 3.0
            why["tcm_embed"] = round(sim, 3)

        # 2.3 西医诊断：结构化 + 文本
        wdx = p.get("western_diagnoses") or []
        apps = f.get("临床应用") or []
        if wdx and apps:
            s_lex = match_score_terms(apps, wdx, w_exact=1.6, w_sub=0.8)
            score += s_lex; why["western"] += round(s_lex,3)
            a = embed_texts(model, ["；".join(wdx)])
            b = embed_texts(model, ["；".join(apps)])
            sim = float(cosine(a,b).max())
            score += sim * 2.2
            why["western"] += round(sim*2.2,3)
        # 从自由文本粗抽关键词再做弱对齐
        kws = extract_keywords(p.get("free_text_dialogue") or "")
        if apps and kws:
            s_lex2 = match_score_terms(apps, kws, w_exact=0.4, w_sub=0.2)
            score += s_lex2; why["lex"] += round(s_lex2,3)

        # 2.4 标签
        tags_face = match_score_terms(f.get("面诊标签") or [], p.get("face_tags") or [], w_exact=0.6, w_sub=0.3)
        tags_tng  = match_score_terms(f.get("舌诊标签") or [], p.get("tongue_tags") or [], w_exact=0.8, w_sub=0.4)
        score += tags_face + tags_tng; why["tags"] = round(tags_face+tags_tng,3)

        # 2.5 自由文本 ↔ 子项文本 的句向量匹配（弱，但覆盖语义）
        subtexts = []
        for pb in pats:
            for sp in (pb.get("subpoints") or []):
                if sp.get("text"): subtexts.append(sp["text"])
        if subtexts:
            a = embed_texts(model, [blob])
            b = embed_texts(model, subtexts[:64])  # 取前64条避免过慢
            sim = float(cosine(a,b).max())
            score += sim * 1.2
            why["text_embed"] = round(sim*1.2,3)

        # 2.6 安全：十八反/十九畏命中直接否决
        fan, wei = conflicts_in_formula(f.get("composition"))
        if fan or wei:
            why["safety"] = fan + wei
            score -= 100

        results.append((score, idx, why))

    results.sort(key=lambda x: x[0], reverse=True)
    top = []
    for sc, idx, why in results[:topk]:
        f = formulas[idx]
        out = {
            "formula_id": f.get("ID"),
            "方名": f.get("方名"),
            "composition": f.get("composition"),
            "出处": f.get("出处"),
            "剂量": f.get("剂量"),
            "剂型": f.get("剂型"),
            "用法": f.get("用法"),
            "功效": f.get("功效"),
            "score": round(float(sc),3),
            "why": why
        }
        top.append(out)
    return top

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formulas", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--index", required=False, help="build_formula_index_v1 生成的 npz；若缺失将临时构建")
    ap.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--recall", type=int, default=30)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    formulas = load_formulas(args.formulas)
    patients = load_patients(args.patients)

    emb_mat, meta = ensure_index(formulas, args.index, args.embed_model)
    model = load_embedder(args.embed_model)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = outdir / "recommendations_v2.jsonl"
    out_csv   = outdir / "topk_preview.csv"

    rows=[]

    with open(out_jsonl, "w", encoding="utf-8") as fo:
        for p in patients:
            top = recommend_for_patient(p, formulas, emb_mat, meta, model, topk=args.topk, recall=args.recall)
            fo.write(json.dumps({"patient_id": p.get("patient_id"), "topk": top}, ensure_ascii=False) + "\n")
            # CSV 预览：只取 top1 的关键字段，方便医生快速浏览
            if top:
                t1 = top[0]
                # 生成一个简短的“成分用量”文本
                comp = t1.get("composition") or []
                comp_txt = "；".join([f"{it.get('药味','')} {it.get('用量_g','')}" for it in comp if it.get("药味")])
                rows.append({
                    "patient_id": p.get("patient_id"),
                    "top1_方名": t1.get("方名"),
                    "top1_功效": t1.get("功效"),
                    "top1_用法": t1.get("用法"),
                    "top1_成分用量": comp_txt[:200]
                })

    import pandas as pd
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"完成：{out_jsonl} ；预览：{out_csv}")

# 复用 build_index 逻辑（如未显式提供索引）
def ensure_index(formulas, index_path, model_name):
    if index_path and Path(index_path).exists():
        return load_index(index_path)
    model = load_embedder(model_name)
    texts=[]
    for f in formulas:
        parts=[f.get("方名","")]
        if f.get("功效"): parts.append("功效："+f["功效"])
        if f.get("临床应用"): parts.append("临床应用："+"；".join(f["临床应用"]))
        for pb in (f.get("patterns") or []):
            seg=[]
            if pb.get("name"): seg.append(pb["name"])
            if pb.get("alias"): seg.append(pb["alias"])
            subs=[sp.get("text","") for sp in (pb.get("subpoints") or []) if sp.get("text")]
            if subs: seg.append("；".join(subs))
            if seg: parts.append("病证："+"。".join(seg))
        if f.get("舌诊标签"): parts.append("舌诊："+"、".join(f["舌诊标签"]))
        if f.get("面诊标签"): parts.append("面诊："+"、".join(f["面诊标签"]))
        texts.append("。".join([p for p in parts if p]))
    emb = embed_texts(model, texts)
    meta = {"id":[f.get("ID") for f in formulas], "name":[f.get("方名") for f in formulas]}
    return emb, meta

if __name__ == "__main__":
    main()
