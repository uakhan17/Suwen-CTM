# -*- coding: utf-8 -*-
# demo_gradio_app_v3_compat_fix_bm25.py
# 在 v3_compat_fix 基础上接入 BM25 融合检索（需要 pip install rank-bm25 jieba）
import argparse, json, re, time, random
from pathlib import Path
import gradio as gr

# ✅ 用带 BM25 融合的包装器替代原 recommend_for_patient
from recommend_bm25_wrapper import recommend_for_patient_with_bm25 as recommend_for_patient

from recommend_v2 import (
    load_formulas, ensure_index, load_embedder
)

EIGHTEEN_FAN = {
    "甘草": ["甘遂","大戟","芫花","海藻"],
    "乌头": ["半夏","瓜蒌","贝母","白蔹","白芨"],
    "藜芦": ["人参","沙参","丹参","玄参","细辛","芍药","苦参"]
}
NINETEEN_WEI = {
    "硫磺": ["朴硝"], "水银": ["砒霜"], "狼毒": ["密陀僧"], "巴豆": ["牵牛"],
    "牙硝": ["三棱"], "丁香": ["郁金"], "肉桂": ["赤石脂"], "人参": ["藜芦"]
}

def parse_tags(s):
    if s is None: return None
    s = s.strip()
    if not s: return None
    parts = re.split(r"[，,、;；\s]+", s)
    parts = [p for p in parts if p]
    return parts or None

def make_patient_blob(free_text, key_symptoms, tcm_syndrome, western_dx, tongue_tags, face_tags):
    return {
        "patient_id": "DEMO",
        "free_text_dialogue": (free_text or "").strip() or None,
        "structured_main_complaint": {
            "duration": None,
            "key_symptoms": parse_tags(key_symptoms) or [],
            "tongue": parse_tags(tongue_tags),
            "face": parse_tags(face_tags),
        },
        "symptoms": parse_tags(key_symptoms) or [],
        "tongue_tags": parse_tags(tongue_tags),
        "face_tags": parse_tags(face_tags),
        "tcm_syndrome": parse_tags(tcm_syndrome),
        "western_diagnoses": parse_tags(western_dx),
    }

def comp_to_markdown(comp):
    if not comp: return "（无组成数据）"
    header = "| 药味 | 用量(g) |\n|---|---|\n"
    rows = []
    for it in comp:
        herb = it.get("药味","")
        dose = it.get("用量_g","")
        rows.append(f"| {herb} | {dose} |")
    return header + "\n".join(rows)

def format_result(topk):
    if not topk:
        return "未检索到推荐结果。"
    md = []
    for i, it in enumerate(topk, 1):
        warn = ""
        if it.get("why",{}).get("safety"):
            pairs = ["×".join(p) for p in it["why"]["safety"]]
            warn = f"\n> ⚠️ **安全警示（疑似十八反/十九畏）**：{ '；'.join(pairs) }"
        md.append(
f"""### Top{i}. {it.get('方名','')}
- **功效**：{it.get('功效','')}
- **出处**：{it.get('出处','')}

**组成**  
{comp_to_markdown(it.get('composition'))}

- **剂量**：{it.get('剂量','')}
- **剂型**：{it.get('剂型','')}
- **用法**：{it.get('用法','')}
- **得分（BM25融合后）**：{it.get('score',0):.4f}

<details>
<summary>展开查看命中理由（patterns / 标签 / 句向量相似度 / BM25）</summary>

- 证候命中：{json.dumps(it.get('why',{}).get('pattern_hits',[]), ensure_ascii=False)}
- 证型相似度（向量）：{it.get('why',{}).get('tcm_embed',0)}
- 西医匹配（词+向量综合）：{it.get('why',{}).get('western',0)}
- 标签匹配：{it.get('why',{}).get('tags',0)}
- 自由文本↔子项（向量）：{it.get('why',{}).get('text_embed',0)}
- BM25分数（归一化）：{it.get('why',{}).get('bm25',0):.4f}
- 融合前总分：{it.get('why',{}).get('score_before_bm25',0):.4f}
- 融合后总分：{it.get('why',{}).get('fused_score',0):.4f}
{warn}
</details>
""")
    return "\n\n---\n\n".join(md)

def conflicts_in_formulas(formulas):
    herbs = []
    for f in formulas:
        for it in (f.get("composition") or []):
            if it.get("药味"): herbs.append(it["药味"].strip())
    herbs_set = set(herbs)
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

def try_load_jsonl(text_or_bytes):
    try:
        s = text_or_bytes.decode("utf-8") if isinstance(text_or_bytes, (bytes, bytearray)) else str(text_or_bytes)
        s = s.strip()
        if not s: return None
        if s.startswith("{"):
            return json.loads(s)
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        cand = [json.loads(ln) for ln in lines if ln.startswith("{")]
        if not cand: return None
        return random.choice(cand)
    except Exception:
        return None

def read_upload_to_bytes(upload_file):
    if upload_file is None:
        return None
    if isinstance(upload_file, (list, tuple)) and upload_file:
        return read_upload_to_bytes(upload_file[0])
    path = getattr(upload_file, "name", None)
    if isinstance(path, str) and Path(path).exists():
        return Path(path).read_bytes()
    if isinstance(upload_file, (str, Path)):
        p = Path(upload_file)
        if p.exists():
            return p.read_bytes()
    read = getattr(upload_file, "read", None)
    if callable(read):
        try:
            return read()
        except Exception:
            pass
    return None

def build_app(formulas_path, index_path, embed_model, recall, topk, share_flag):
    formulas = load_formulas(formulas_path)
    emb_mat, meta = ensure_index(formulas, index_path, embed_model)
    model = load_embedder(embed_model)

    last_topk_cache = {"items": []}

    def infer(free_text, key_symptoms, tcm_syndrome, western_dx, tongue_tags, face_tags, 
              recall_ui, topk_ui, recall_bm25_ui, bm25_weight_ui):
        t0 = time.time()
        p = make_patient_blob(free_text, key_symptoms, tcm_syndrome, western_dx, tongue_tags, face_tags)
        top = recommend_for_patient(
            p, formulas, emb_mat, meta, model,
            topk=int(topk_ui), recall=int(recall_ui),
            recall_bm25=int(recall_bm25_ui), bm25_weight=float(bm25_weight_ui)
        )
        md = format_result(top)
        last_topk_cache["items"] = top
        t1 = time.time()
        return md, json.dumps({"patient": p, "topk": top}, ensure_ascii=False, indent=2), f"{t1-t0:.2f}s"

    def load_random_sample(upload_file, default_path):
        patient = None
        data = read_upload_to_bytes(upload_file)
        if data is not None:
            patient = try_load_jsonl(data)
        if patient is None and default_path:
            p = Path(default_path)
            if p.exists():
                lines = [ln.strip() for ln in p.read_text("utf-8").splitlines() if ln.strip()]
                objs = [json.loads(ln) for ln in lines if ln.startswith("{")]
                if objs:
                    import random as _r
                    patient = _r.choice(objs)
        if patient is None:
            return gr.update(), "", "", "", "", "", "未找到可导入的JSON/JSONL，请上传文件或指定默认路径。"
        free_text = patient.get("free_text_dialogue") or ""
        ks = patient.get("structured_main_complaint",{}).get("key_symptoms") or patient.get("symptoms") or []
        key_symptoms = "，".join(ks)
        tcm = "，".join(patient.get("tcm_syndrome") or [])
        wdx = "，".join(patient.get("western_diagnoses") or [])
        ttags = "，".join(patient.get("tongue_tags") or [])
        ftags = "，".join(patient.get("face_tags") or [])
        return gr.update(), free_text, key_symptoms, tcm, wdx, ttags, ftags

    def load_conflict_case(conflict_path):
        try:
            p = Path(conflict_path)
            if not p.exists():
                return "", "", "", "", "", "", f"未找到 {conflict_path}"
            patient = json.loads(p.read_text("utf-8"))
            free_text = patient.get("free_text_dialogue") or ""
            ks = patient.get("structured_main_complaint",{}).get("key_symptoms") or patient.get("symptoms") or []
            key_symptoms = "，".join(ks)
            tcm = "，".join(patient.get("tcm_syndrome") or [])
            wdx = "，".join(patient.get("western_diagnoses") or [])
            ttags = "，".join(patient.get("tongue_tags") or [])
            ftags = "，".join(patient.get("face_tags") or [])
            return free_text, key_symptoms, tcm, wdx, ttags, ftags, f"已载入：{conflict_path}"
        except Exception as e:
            return "", "", "", "", "", "", f"载入失败：{e}"

    def run_gflops_bench(n, device):
        try:
            import torch, time as _time
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dev = torch.device(device)
            a = torch.randn(n, n, dtype=torch.float32, device=dev)
            b = torch.randn(n, n, dtype=torch.float32, device=dev)
            for _ in range(3):
                _ = a @ b
                if dev.type == "cuda": torch.cuda.synchronize()
            iters = 5
            t0 = _time.time()
            for _ in range(iters):
                _ = a @ b
                if dev.type == "cuda": torch.cuda.synchronize()
            t1 = _time.time()
            avg = (t1 - t0) / iters
            flops = 2 * (n**3)
            gflops = flops / avg / 1e9
            devname = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
            return f"{gflops:.1f} GFLOP/s @ {devname}  (n={n})"
        except Exception as e:
            return f"无法运行基准：{e}"

    def check_combo(idx1, idx2):
        try:
            i1 = int(idx1); i2 = int(idx2)
        except:
            return "请输入要合方的 Top 序号（例如 1 和 2）。"
        arr = last_topk_cache.get("items") or []
        if not arr:
            return "请先运行一次推荐，再进行合方安全检查。"
        sel = []
        for i in [i1-1, i2-1]:
            if 0 <= i < len(arr): sel.append(arr[i])
        if len(sel) < 2:
            return "当前推荐数量不足或序号不正确。"
        fan, wei = conflicts_in_formulas(sel)
        if not fan and not wei:
            return "✅ 合方安全：未检出十八反/十九畏禁配。"
        msg = []
        if fan: msg.append("**十八反**：" + "；".join(["×".join(x) for x in fan]))
        if wei: msg.append("**十九畏**：" + "；".join(["×".join(x) for x in wei]))
        return "⚠️ 合方存在禁配风险：\n" + "\n".join(msg)

    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("# 💊 方剂推荐系统")
        gr.Markdown("两阶段检索：BM25 召回 + 句向量召回 → 融合重排")

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 一键导入病例")
                upload = gr.File(label="上传 JSON / JSONL", file_count="single", file_types=[".json",".jsonl",".txt"])
                default_path = gr.Textbox(label="默认 JSONL 路径", value="patients_v2.jsonl", placeholder="不上传则尝试此路径并随机抽取一例")
                btn_import = gr.Button("📥 随机载入病例到输入框", variant="secondary")
                conflict_path = gr.Textbox(label="冲突案例", value="example_case.json")
                btn_conflict = gr.Button("⚡ 载入示例（可判断十八反/十九畏）", variant="secondary")
                import_status = gr.Markdown("")

                free_text = gr.Textbox(label="自由文本（医患对话/专业摘要）", lines=6)
                key_symptoms = gr.Textbox(label="结构化主诉：症状（逗号/顿号分隔）")
                tcm_syndrome = gr.Textbox(label="中医证型（逗号/顿号分隔）")
                western_dx = gr.Textbox(label="西医诊断（逗号/顿号分隔）")
                tongue_tags = gr.Textbox(label="舌诊标签（逗号/顿号分隔）")
                face_tags = gr.Textbox(label="面诊标签（逗号/顿号分隔）")

                with gr.Row():
                    recall_ui = gr.Slider(10, 100, value=30, step=1, label="向量召回候选数（dense）")
                    topk_ui = gr.Slider(1, 10, value=5, step=1, label="输出Top-K")
                with gr.Row():
                    recall_bm25_ui = gr.Slider(10, 200, value=30, step=1, label="BM25 召回候选数（sparse）")
                    bm25_weight_ui = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="融合权重（BM25 权重）")
                btn = gr.Button("🚀 生成推荐（BM25+向量 融合）", variant="primary")

                with gr.Accordion("性能与安全工具", open=False):
                    with gr.Row():
                        n_mat = gr.Slider(256, 2048, value=1024, step=128, label="GFLOPS基准矩阵维度 n")
                        device_sel = gr.Radio(["auto","cpu","cuda"], value="auto", label="设备")
                        btn_gflops = gr.Button("🏎️ 运行GFLOPS基准", variant="secondary")
                        gflops_out = gr.Textbox(label="GFLOPS", interactive=False)
                    with gr.Row():
                        idx1 = gr.Textbox(label="合方：Top序号1", value="1")
                        idx2 = gr.Textbox(label="合方：Top序号2", value="2")
                        btn_combo = gr.Button("🛡️ 合方安全检查", variant="secondary")
                        combo_out = gr.Markdown("（先生成推荐，再检查合方安全）")

            with gr.Column(scale=5):
                out_md = gr.Markdown(value="（结果将在这里显示）")
                out_json = gr.Code(label="原始JSON（便于联调/保存）")
                latency = gr.Textbox(label="耗时", interactive=False)
                gr.Markdown("> 免责声明：本演示仅用于产品与技术验证，不构成医疗建议。")

        btn.click(
            fn=infer,
            inputs=[free_text, key_symptoms, tcm_syndrome, western_dx, tongue_tags, face_tags, 
                    recall_ui, topk_ui, recall_bm25_ui, bm25_weight_ui],
            outputs=[out_md, out_json, latency]
        )
        btn_import.click(
            fn=load_random_sample,
            inputs=[upload, default_path],
            outputs=[upload, free_text, key_symptoms, tcm_syndrome, western_dx, tongue_tags, face_tags]
        )
        btn_conflict.click(
            fn=load_conflict_case,
            inputs=[conflict_path],
            outputs=[free_text, key_symptoms, tcm_syndrome, western_dx, tongue_tags, face_tags, import_status]
        )
        btn_gflops.click(
            fn=run_gflops_bench,
            inputs=[n_mat, device_sel],
            outputs=[gflops_out]
        )
        btn_combo.click(
            fn=check_combo,
            inputs=[idx1, idx2],
            outputs=[combo_out]
        )
    return demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--formulas", required=True)
    ap.add_argument("--index", required=False, default=None)
    ap.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--recall", type=int, default=30)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--server-port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="创建公网可访问的临时链接（gradio.live）")
    args = ap.parse_args()

    app = build_app(args.formulas, args.index, args.embed_model, args.recall, args.topk, args.share)
    app.launch(server_name="0.0.0.0", server_port=args.server_port, inbrowser=False, share=args.share)
