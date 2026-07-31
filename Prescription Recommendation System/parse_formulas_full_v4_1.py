#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_formulas_full_v4_1.py
- 更强的“主治病证”解析（半角/全角统一、支持 1. / 1． / 1、）
- 其它逻辑沿用 v4（注意为全方级；“无”→ null；composition 仅输出存在列）
"""
import argparse, json, math, re, unicodedata
from pathlib import Path
import pandas as pd

# ---------- helpers ----------
def to_halfwidth(s: str) -> str:
    # 全角 -> 半角（包括数字和标点）
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return unicodedata.normalize("NFKC", s)

def strip_space(s: str) -> str:
    return to_halfwidth(s).replace("\u3000"," ").strip()

def read_csv_flexible(path: Path):
    for enc in ["utf-8-sig","utf-8","gbk","gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except Exception:
            continue
    return pd.read_csv(path, encoding_errors="ignore"), "auto"

def find_col(df, candidates):
    orig_cols = list(df.columns)
    def norm(x): return re.sub(r"\s+","", to_halfwidth(str(x)).lower())
    pool = {norm(c): c for c in orig_cols}
    for cand in candidates:
        k = norm(cand)
        if k in pool: return pool[k]
    return None

def is_literal_none(val: str) -> bool:
    if val is None: return False
    v = strip_space(val).lower()
    return v in {"无","none","null","n/a","na"}

def split_tokens(s: str):
    if s is None: return []
    s = strip_space(s)
    if not s: return []
    parts = re.split(r"[，,；;、/]\s*", s)
    return [p for p in (x.strip() for x in parts) if p]

def tokens_or_null(s: str):
    if s is None or str(s).strip()=="": return None
    if is_literal_none(s): return None
    toks = split_tokens(s)
    return toks if toks else None

def text_or_null(s: str):
    if s is None: return None
    s2 = strip_space(s)
    return None if (not s2 or is_literal_none(s2)) else s2

def to_number(x):
    if x is None: return ""
    xs = strip_space(x)
    if not xs: return ""
    m = re.search(r"(\d+(?:\.\d+)?)", xs)
    if m:
        try: return float(m.group(1))
        except Exception: return m.group(1)
    return xs

def parse_composition(row, col_ing, col_dose, col_prep=None, col_note=None):
    herbs = split_tokens(row.get(col_ing, "")) if col_ing else []
    doses = split_tokens(row.get(col_dose, "")) if col_dose else []
    preps = split_tokens(row.get(col_prep, "")) if col_prep else []
    notes = split_tokens(row.get(col_note, "")) if col_note else []
    L = max(len(herbs), len(doses), len(preps), len(notes))
    comp = []
    for i in range(L):
        herb = herbs[i] if i < len(herbs) else ""
        dose = to_number(doses[i]) if i < len(doses) else ""
        item = {"药味": herb, "用量_g": dose}
        if col_prep: item["炮制用法"] = preps[i] if i < len(preps) else ""
        if col_note: item["备注"]     = notes[i] if i < len(notes) else ""
        if any(v != "" for v in item.values()): comp.append(item)
    return comp, (len(herbs), len(doses), len(preps), len(notes))

# ---------- patterns parsing (stronger) ----------
def parse_top_sections(text: str):
    """
    支持 1. / 1． / 1、 作为分块起始；半角/全角统一后再解析；
    在分块内部，优先用第一处冒号 : 分割标题与正文；否则用第一个句读符分割；找不到则正文置空。
    """
    s = strip_space(text)
    if not s: return []
    # 允许 '1.' '1．' '1、'
    it = list(re.finditer(r'(?<!\d)(\d{1,3})[\.．、]\s*', s))
    if not it:
        return [("1","未命名病证", s)]
    sections = []
    for idx, m in enumerate(it):
        code = m.group(1)
        start = m.end()
        end = it[idx+1].start() if idx+1 < len(it) else len(s)
        chunk = s[start:end].strip()
        # 拆 title/body
        mcol = re.search(r'[:：]', chunk)
        if mcol:
            title = strip_space(chunk[:mcol.start()])
            body  = strip_space(chunk[mcol.end():])
        else:
            msep = re.search(r'[。.;；;，,]', chunk)
            if msep:
                title = strip_space(chunk[:msep.start()])
                body  = strip_space(chunk[msep.end():])
            else:
                title, body = chunk, ""
        sections.append((code, title, body))
    return sections

def extract_inline_rule(title: str):
    m = re.search(r'\{\s*(ALL|ANY|K\s*=\s*\d+)\s*\}\s*$', title, flags=re.IGNORECASE)
    if not m: return title, None
    tag = m.group(1).strip().upper()
    clean = title[:m.start()].strip()
    if tag == "ALL": return clean, {"type":"all"}
    if tag == "ANY": return clean, {"type":"any"}
    km = re.match(r'K\s*=\s*(\d+)', tag)
    if km: return clean, {"type":"k_of_n","k":int(km.group(1))}
    return clean, None

def split_title(title_with_paren: str):
    m = re.match(r'^(.*?)[（(]([^（）()]*)[）)]\s*$', title_with_paren.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return title_with_paren.strip(), ""

def parse_subpoints(body: str, top_code: str):
    s = strip_space(body).replace("；",";")
    # 接受形如 1a. / 1a。 / 1a（无点）
    pat = re.compile(rf'({re.escape(top_code)}[a-zA-Z])\s*[\.。]?\s*')
    items = []
    last = 0; last_code = None
    for m in pat.finditer(s):
        if last_code is not None:
            items.append((last_code, s[last:m.start()].strip(" ;；")))
        last_code = m.group(1)
        last = m.end()
    if last_code is not None:
        items.append((last_code, s[last:].strip(" ;；")))
    else:
        if s:
            items.append((f"{top_code}a", s))
    return [{"code": c, "text": t} for c,t in items]

def make_rule(default_rule: str, n: int, k: int|None=None, override: dict|None=None):
    if override:
        if override.get("type") == "k_of_n":
            kk = override.get("k") or ((n+1)//2 if n>0 else 1)
            return {"type":"k_of_n","k":int(kk),"n":int(n)}
        return override
    dr = (default_rule or "k-of-n").lower()
    if dr=="any": return {"type":"any"}
    if dr=="all": return {"type":"all"}
    kk = int(k) if k else ((n+1)//2 if n>0 else 1)
    return {"type":"k_of_n","k":kk,"n":int(n)}

# ---------- main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--default-rule", dest="default_rule", default="k-of-n", choices=["any","all","k-of-n"])
    ap.add_argument("--k", dest="k", type=int, default=None)
    # columns
    ap.add_argument("--col-id", default=None)
    ap.add_argument("--col-name", default=None)
    ap.add_argument("--col-source", default=None)
    ap.add_argument("--col-efficacy", default=None)
    ap.add_argument("--col-components", default=None)
    ap.add_argument("--col-doses", default=None)
    ap.add_argument("--col-prep", default=None)
    ap.add_argument("--col-note", default=None)
    ap.add_argument("--col-totaldose", default=None)
    ap.add_argument("--col-dosageform", default=None)
    ap.add_argument("--col-usage", default=None)
    ap.add_argument("--col-face-tags", default=None)
    ap.add_argument("--col-tongue-tags", default=None)
    ap.add_argument("--col-symptoms", default=None)
    ap.add_argument("--col-applications", default=None)
    ap.add_argument("--col-adjust", default=None)
    ap.add_argument("--col-patterns", default=None)
    ap.add_argument("--col-notice", default=None)
    args = ap.parse_args()

    df, _ = read_csv_flexible(Path(args.inp))

    # column detection
    col_id       = args.col_id        or find_col(df, ["ID","编号","序号"])
    col_name     = args.col_name      or find_col(df, ["方名","方剂","处方名","方剂名","方"])
    col_source   = args.col_source    or find_col(df, ["出处","来源"])
    col_efficacy = args.col_efficacy  or find_col(df, ["功效","主功效","功用"])
    col_ing      = args.col_components or find_col(df, ["成分","组成","药味","药味列表"])
    col_dose     = args.col_doses     or find_col(df, ["用量","剂量表","克数","用量(g)","用量_g","每味用量"])
    col_prep     = args.col_prep      or find_col(df, ["炮制用法","炮制","处理"])
    col_note     = args.col_note      or find_col(df, ["备注","说明"])
    col_total    = args.col_totaldose or find_col(df, ["剂量","总剂量"])
    col_form     = args.col_dosageform or find_col(df, ["剂型"])
    col_usage    = args.col_usage     or find_col(df, ["用法","煎服","煎服要点","服法"])
    col_face     = args.col_face_tags or find_col(df, ["面诊标签","面诊"])
    col_tongue   = args.col_tongue_tags or find_col(df, ["舌诊标签","舌诊"])
    col_symp     = args.col_skeleton if hasattr(args,"col_skeleton") and args.col_skeleton else (args.col_symptoms or find_col(df, ["主治症状","症状"]))
    col_apps     = args.col_applications or find_col(df, ["临床应用","现代应用","现代主治"])
    col_adjust   = args.col_adjust    or find_col(df, ["加味/去味","加减","加味去味"])
    col_patterns = args.col_patterns  or find_col(df, ["主治病证","主治病症","病证","主治"])
    col_notice   = args.col_notice    or find_col(df, ["注意","注意事项"])

    records = []
    for i, row in df.iterrows():
        rid   = text_or_null(row.get(col_id, "")) if col_id else None
        name  = text_or_null(row.get(col_name, "")) if col_name else None
        if not name: name = f"未命名方剂_{i+1}"
        src   = text_or_null(row.get(col_source, "")) if col_source else None
        eff   = text_or_null(row.get(col_efficacy, "")) if col_efficacy else None

        composition = []
        if col_ing and col_dose:
            comp, lens = parse_composition(row, col_ing, col_dose, col_prep, col_note)
            composition = comp

        totaldose = text_or_null(row.get(col_total, "")) if col_total else None
        dosageform= text_or_null(row.get(col_form, "")) if col_form else None
        usage     = text_or_null(row.get(col_usage, "")) if col_usage else None
        face_tags   = tokens_or_null(row.get(col_face, "")) if col_face else None
        tongue_tags = tokens_or_null(row.get(col_tongue, "")) if col_tongue else None
        symptoms    = tokens_or_null(row.get(col_symp, "")) if col_symp else None
        applications= tokens_or_null(row.get(col_apps, "")) if col_apps else None
        notice      = text_or_null(row.get(col_notice, "")) if col_notice else None

        adjust = None
        if col_adjust:
            raw = strip_space(row.get(col_adjust, ""))
            if raw and not is_literal_none(raw):
                def parse_adj(t: str):
                    res = {"add": [], "remove": []}
                    for sign, item in re.findall(r"([+\-])\s*([^+\-；;，,]+)", t):
                        toks = split_tokens(item)
                        (res["add"] if sign=="+" else res["remove"]).extend(toks or [item.strip()])
                    for label, key in [("加味","add"),("加","add"),("去味","remove"),("去","remove")]:
                        m = re.search(rf"{label}\s*[:：]\s*([^；;]+)", t)
                        if m: res[key].extend(split_tokens(m.group(1)))
                    res["add"]    = [x for i,x in enumerate(res["add"])    if x and x not in res["add"][:i]]
                    res["remove"] = [x for i,x in enumerate(res["remove"]) if x and x not in res["remove"][:i]]
                    return res if (res["add"] or res["remove"]) else None
                adjust = parse_adj(raw)

        patterns = None; raw_patterns = None
        if col_patterns:
            raw_patterns = strip_space(row.get(col_patterns, ""))
            if raw_patterns and not is_literal_none(raw_patterns):
                tops = parse_top_sections(raw_patterns)
                pat_list = []
                for code, title_with_paren, body in tops:
                    title_with_mark, ov_rule = extract_inline_rule(title_with_paren)
                    title, alias = split_title(title_with_mark)
                    subs = parse_subpoints(body, code)
                    rule = make_rule(args.default_rule, len(subs), args.k, ov_rule)
                    pat_list.append({"code": code, "name": title, "alias": alias, "rule": rule, "subpoints": subs})
                patterns = pat_list

        rec = {"方名": name, "composition": composition}
        if rid is not None: rec["ID"] = rid
        if src is not None: rec["出处"] = src
        if eff is not None: rec["功效"] = eff
        if totaldose is not None: rec["剂量"] = totaldose
        if dosageform is not None: rec["剂型"] = dosageform
        if usage is not None: rec["用法"] = usage
        if face_tags is not None: rec["面诊标签"] = face_tags
        if tongue_tags is not None: rec["舌诊标签"] = tongue_tags
        if symptoms is not None: rec["主治症状"] = symptoms
        if applications is not None: rec["临床应用"] = applications
        if notice is not None: rec["注意"] = notice
        if adjust is not None: rec["加味去味"] = adjust
        if raw_patterns: rec["主治病证_raw"] = raw_patterns
        if patterns is not None: rec["patterns"] = patterns
        records.append(rec)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "方剂v4_1_完整.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(str(jsonl_path))
if __name__ == "__main__":
    main()
