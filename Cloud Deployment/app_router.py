# -*- coding: utf-8 -*-
import os, io, base64, math, hashlib
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F  # noqa: F401 (kept for parity / potential custom heads)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import numpy as np
import cv2

# ---------------------- Config: paths ----------------------
SCRFD_ONNX = os.getenv("SCRFD_ONNX", "./scrfd_2.5g_bnkps.onnx")

# Inference knobs (env-overridable)
MIN_PROB = float(os.getenv("MIN_PROB", "0.10"))
INFER_TTA = int(os.getenv("INFER_TTA", "0")) == 1
PROB_CAP = float(os.getenv("PROB_CAP", "0.85"))

UNCERTAINTY_METRIC = os.getenv("UNCERTAINTY_METRIC", "margin")  # 'margin' | 'entropy'
MARGIN_TH = float(os.getenv("MARGIN_TH", "1.0"))
ENTROPY_TH = float(os.getenv("ENTROPY_TH", "1.10"))

EXPLORE_ON = int(os.getenv("EXPLORE_ON", "1")) == 1
EXPLORE_MODE = os.getenv("EXPLORE_MODE", "full")  # 'lock1' | 'full'
TOPN_POOL = int(os.getenv("TOPN_POOL", "7"))
SAMPLE_T = float(os.getenv("SAMPLE_T", "1.5"))
GUMBEL_EPS = float(os.getenv("GUMBEL_EPS", "1e-9"))
SEED_SALT = os.getenv("SEED_SALT", "ctm_salt_v1")

# Optional: allow small auto-rotate for face gate only
AUTO_ROTATE_MAX_DEG = float(os.getenv("AUTO_ROTATE_MAX_DEG", "0"))

# ---------------------- Third-party gates ----------------------
from audit_face import load_scrfd as load_scrfd_face, gate_face_one
import audit_tongue  # expect DET, PROVIDERS are lazy (None) and set in lifespan()

# ---------------------- Your multitask model ----------------------
from model import MultiTaskNet

# ---------------------- FastAPI app (lifespan) ----------------------
app = FastAPI(title="CTM Face & Tongue Inference")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Labels / task order ----------------------
FACE_LABELS: Dict[str, List[str]] = {
    '望眼_目色': ['目胞色黑晦暗', '白睛发黄', '两眦淡白', '目赤肿痛'],
    '望眼_目态': ['目睛凝视', '胞睑下垂', '睡眠露睛', '目睛上视', '斜视'],
    '望眼_目形': ['眼球凹陷', '眼球突出', '胞睑红肿', '目胞浮肿', '无异常'],
    '望眼_瞳孔': ['瞳孔缩小', '瞳孔等大', '瞳孔散大'],
    '望口_口形': ['口角无异常', '口歪不收', '口疮', '口糜', '鹅口疮'],
    '望口_口态': ['口张', '口噤', '口撮', '口', '口振', '口动', '无异常'],
    '望唇_唇色': ['淡白', '深红', '赤红', '樱桃红', '青紫', '青黑', '红润'],
    '望唇_唇形': ['无异常', '唇干而裂', '嘴唇糜烂', '唇内溃烂', '唇边生疮', '唇角生疔', '口唇翻卷不能覆齿'],
    '望鼻_鼻色': ['色白', '色赤', '微黄', '灰暗枯槁', '红黄隐隐 明润含蓄'],
    '望鼻_鼻形': ['鼻头生疖', '生粉刺', '鼻翼扇动', '鼻柱溃陷', '无异常'],
    '面色_面色': ['青黄','红赤满面通红','午后潮红','久病苍白却时颧赤泛红如妆','淡白无华',
                 '晄白','苍白','淡青','青黑','青灰','青黄','黧黑晦暗','紫暗黧黑','黑而干焦',
                 '眼眶发黑','萎黄','黄胖','阳黄','阴黄','面黄','红黄隐隐明润含蓄'],
    '面色_皮肤光泽': ['荣润','枯槁'],
    '面色_面形': ['无异常','面肿','腮肿','面削颧耸','口眼斜','惊恐貌','苦笑貌'],
}
FACE_TASK_ORDER: List[str] = [
    '望眼_目色','望眼_目态','望眼_目形','望眼_瞳孔',
    '望口_口形','望口_口态',
    '望唇_唇色','望唇_唇形',
    '望鼻_鼻色','望鼻_鼻形',
    '面色_面色','面色_皮肤光泽','面色_面形',
]

# ---------------------- TONGUE labels / order ----------------------
TONGUE_LABELS: Dict[str, List[str]] = {
    '舌质_神'   : ['枯舌', '荣舌'],
    '舌质_色'   : ['淡红', '红', '淡白', '绛红', '青紫'],
    '舌质_形'   : ['老', '嫩', '胖', '瘦', '齿痕', '点刺', '裂纹'],
    '舌质_态'   : ['痿软', '强硬', '歪斜', '颤动', '吐弄', '短缩'],
    '舌苔_苔色' : ['灰黑', '白', '黄', '无', '滑', '少'],
    '舌苔_苔质' : ['燥', '薄', '厚', '润', '腻', '腐', '剥落', '偏全'],
}
TONGUE_TASK_ORDER: List[str] = [
    '舌质_神','舌质_色','舌质_形','舌质_态','舌苔_苔色','舌苔_苔质'
]

# ---------------------- Preprocess ----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _pil_bytes_to_tensor(b: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    t = torch.as_tensor(np.array(img), dtype=torch.uint8).permute(2,0,1).contiguous()
    return t

def _preprocess_tensor(x_u8: torch.Tensor, size: int) -> torch.Tensor:
    x = x_u8.float() / 255.0
    x = TF.resize(x, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return x

# ---------------------- Model pack ----------------------
class _ModelPack:
    def __init__(self, labels: Dict[str, List[str]], task_order: List[str],
                 weights: str, default_backbone="convnext_nano_ols.d1h_in1k", default_size=224):
        self.labels = labels
        self.task_order = task_order
        self.size = default_size
        self.backbone = default_backbone

        blob = torch.load(weights, map_location=DEVICE)
        state_dict = blob.get("model_ema") or blob.get("model") or blob
        self.backbone = blob.get("backbone", self.backbone)
        self.size = int(blob.get("size", self.size))

        num_classes = {k: len(self.labels[k]) for k in self.task_order}
        self.model = MultiTaskNet(num_classes_dict=num_classes,
                                  backbone=self.backbone,
                                  pretrained=False,
                                  dropout=0.0).to(DEVICE).eval()
        self.model.load_state_dict(state_dict, strict=True)

        with torch.no_grad():
            dummy = torch.zeros(1,3,self.size,self.size, device=DEVICE)
            _ = self.model(dummy)

MODELS: Dict[str, _ModelPack] = {}

# ---------------------- Inference helpers ----------------------
def _logit_margin(z: torch.Tensor) -> float:
    v, _ = torch.sort(z, descending=True)
    if v.numel() < 2:
        return float('inf')
    return float(v[0] - v[1])

def _entropy_from_logits(z: torch.Tensor) -> float:
    p = z.softmax(-1).clamp_min(1e-12)
    return float(-(p * p.log()).sum())

def _is_uncertain(z: torch.Tensor) -> bool:
    if UNCERTAINTY_METRIC == "entropy":
        return _entropy_from_logits(z) > ENTROPY_TH
    return _logit_margin(z) < MARGIN_TH

def _hash_uniform_0_1(key: str) -> float:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (int(h, 16) % (10**12)) / 1e12

def _gumbel_noise_from_key(key: str, C: int) -> torch.Tensor:
    vals = []
    for j in range(C):
        u = max(_hash_uniform_0_1(f"{key}|{j}"), GUMBEL_EPS)
        g = -math.log(-math.log(u))
        vals.append(g)
    return torch.tensor(vals, dtype=torch.float32, device=DEVICE)

def _forward_to_dict(pack: _ModelPack, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = pack.model(x)
    if isinstance(out, dict):
        return out
    if isinstance(out, (list, tuple)):
        return {head: out[i] for i, head in enumerate(pack.task_order)}
    raise TypeError(f"Unsupported model output type: {type(out)}")

@torch.no_grad()
def _infer_one(pack: _ModelPack, x_u8: torch.Tensor, topk: int, request_id: str) -> Dict[str, Any]:
    x = _preprocess_tensor(x_u8, pack.size).unsqueeze(0).to(DEVICE, non_blocking=True)

    if INFER_TTA:
        x2 = torch.flip(x, dims=[3])
        out1 = _forward_to_dict(pack, x)
        out2 = _forward_to_dict(pack, x2)
        logits_dict = {h: (out1[h] + out2[h]) * 0.5 for h in pack.task_order}
    else:
        logits_dict = _forward_to_dict(pack, x)

    out: Dict[str, Any] = {}
    for head in pack.task_order:
        logits = logits_dict[head].squeeze(0)  # [C]
        probs = logits.softmax(-1)
        if PROB_CAP > 0.0:
            p = torch.minimum(probs, torch.tensor(PROB_CAP, device=probs.device))
            probs = p / p.sum()

        C = probs.numel()
        kk = min(topk, C)
        do_explore = EXPLORE_ON and _is_uncertain(logits)

        if not do_explore:
            vals, idxs = torch.sort(probs, descending=True)
            mask = vals >= MIN_PROB
            sel = idxs[mask][:kk]
            if sel.numel() < kk:
                need = kk - sel.numel()
                fill = idxs[~mask][:need]
                sel = torch.cat([sel, fill], dim=0)
            chosen = sel
        else:
            vals, idxs = torch.sort(probs, descending=True)
            pool = idxs[:max(TOPN_POOL, kk)].clone()
            log_p = probs.clamp_min(1e-12).log()
            chosen = _sample_topk_from_pool(
                log_p, kk, request_id=request_id, head=head, idx_pool=pool, mode=EXPLORE_MODE
            )

        labels = pack.labels[head]
        out[head] = [{"label": labels[int(i)], "prob": round(float(probs[int(i)]), 4)}
                     for i in chosen.tolist()]
    return out

def _sample_topk_from_pool(log_p: torch.Tensor,
                           k: int,
                           request_id: str,
                           head: str,
                           idx_pool: Optional[torch.Tensor] = None,
                           mode: str = "lock1") -> torch.Tensor:
    if idx_pool is None:
        C = log_p.numel()
        idx_pool = torch.arange(C, device=DEVICE)
    if mode == "full":
        g = _gumbel_noise_from_key(f"{SEED_SALT}|{request_id}|{head}", idx_pool.numel())
        return torch.topk((log_p[idx_pool] / max(SAMPLE_T,1e-6)) + g, k=k, dim=-1).indices
    vals, idxs = torch.sort(log_p, descending=True)
    sel = [int(idxs[0].item())]
    rest = idxs[1:max(TOPN_POOL, k)]
    g = _gumbel_noise_from_key(f"{SEED_SALT}|{request_id}|{head}", rest.numel())
    take = torch.topk((log_p[rest] / max(SAMPLE_T,1e-6)) + g, k=min(k-1, rest.numel())).indices
    sel.extend([int(rest[i].item()) for i in take.tolist()])
    return torch.tensor(sel, dtype=torch.long, device=DEVICE)

# ---------------------- Gate helpers ----------------------
def _to_bgr_np(x_u8: torch.Tensor) -> np.ndarray:
    arr = x_u8.detach().cpu().numpy()
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError("expecting [3,H,W] uint8 tensor")
    img_rgb = np.transpose(arr, (1, 2, 0))
    return img_rgb[..., ::-1].copy()

def _run_gate(model_name: str, x_u8: torch.Tensor):
    bgr = _to_bgr_np(x_u8)
    if model_name == "face":
        res = gate_face_one(FACE_DET, bgr, use_fallback=True)
        return bool(res.get("ok", False)), str(res.get("reason", "")), res
    elif model_name == "tongue":
        ok, msg, info = audit_tongue.gate_tongue(bgr)
        return bool(ok), str(msg), info
    else:
        return True, "OK", {}

def _rotate_bgr_center(bgr: np.ndarray, deg: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _maybe_autofix_and_retry_gate(model: str, x_u8: torch.Tensor, gate_ok: bool, gate_info: dict):
    # Only for face; allow small auto-rotate retry if configured
    if gate_ok or model != "face":
        return gate_ok, gate_info.get("reason", ""), gate_info, x_u8, False
    roll = gate_info.get("roll_deg", None)
    nfaces = gate_info.get("num_faces", gate_info.get("n_faces", None))
    if (nfaces == 1) and (roll is not None) and (abs(float(roll)) <= AUTO_ROTATE_MAX_DEG):
        bgr = _to_bgr_np(x_u8)
        bgr_fix = _rotate_bgr_center(bgr, -float(roll))
        x_fix = torch.from_numpy(bgr_fix[..., ::-1].copy()).permute(2,0,1).contiguous().to(dtype=torch.uint8)
        ok2, msg2, info2 = _run_gate(model, x_fix)
        if ok2:
            info2 = dict(info2 or {})
            info2["auto_fix"] = {"op": "rotate", "deg": float(-float(roll))}
            return True, "", info2, x_fix, True
    return False, gate_info.get("reason", ""), gate_info, x_u8, False

# ---------------------- Request-ID guard ----------------------
def _require_request_id(req: Request, req_id: Optional[str]) -> str:
    if not req_id or not isinstance(req_id, str) or len(req_id) < 3:
        raise HTTPException(400, "missing or invalid request_id")
    return req_id

# ---------------------- Lifespan: init ----------------------
FACE_DET = None
FACE_DET_PROVIDERS = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init models
    face_w = os.getenv("FACE_WEIGHTS", "weights_face.pt")
    tongue_w = os.getenv("TONGUE_WEIGHTS", "weights_tongue.pt")
    MODELS["face"]   = _ModelPack(FACE_LABELS, FACE_TASK_ORDER, face_w)
    MODELS["tongue"] = _ModelPack(TONGUE_LABELS, TONGUE_TASK_ORDER, tongue_w)

    # Init SCRFD once; share with both gates
    global FACE_DET, FACE_DET_PROVIDERS
    FACE_DET, FACE_DET_PROVIDERS = load_scrfd_face(SCRFD_ONNX, use_gpu=torch.cuda.is_available())
    # Make audit_tongue use same detector (ensure audit_tongue is lazy)
    audit_tongue.DET, audit_tongue.PROVIDERS = audit_tongue.load_scrfd(
        onnx_path=SCRFD_ONNX, use_gpu=torch.cuda.is_available()
    )

    print(f"[Startup] models loaded. SCRFD={SCRFD_ONNX} providers_face={FACE_DET_PROVIDERS}")
    try:
        yield
    finally:
        print("[Shutdown] done")

app.router.lifespan_context = lifespan  # attach lifespan to app

# ---------------------- Core helpers for strict gate behavior ----------------------
def _reject_payload(rid: str, model: str, reason: str, info: dict) -> Dict[str, Any]:
    # HARD REJECT: empty results + Chinese warning
    return {
        "request_id": rid,
        "model_used": model,
        "gate": {"ok": False, "reason": reason, "info": info},
        "warning": "采集信息有误，请重拍照片",
        "results": []
    }

def _accept_payload(rid: str, model: str, info: dict, results: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "request_id": rid,
        "model_used": model,
        "gate": {"ok": True, "info": info},
        "results": results
    }

# ---------------------- Main endpoint (/predict) ----------------------
@app.post("/predict")
async def predict(request: Request,
                  file: UploadFile | None = File(None),
                  model: str = Form("face"),
                  topk: int = Form(3),
                  request_id: str = Form(None),
                  proceed: str = Form(None),           # kept for compatibility; ignored
                  confirm_token: str = Form(None)):    # kept for compatibility; ignored
    # A) Form-data upload path
    if file is not None:
        rid = _require_request_id(request, request_id)
        if model not in MODELS:
            raise HTTPException(400, f"model must be one of {list(MODELS.keys())}")
        img_bytes = await file.read()
        x_u8 = _pil_bytes_to_tensor(img_bytes)

        ok, msg, info = _run_gate(model, x_u8)
        ok2, msg2, info2, x_u8_new, fixed = _maybe_autofix_and_retry_gate(model, x_u8, ok, info or {})
        if fixed:
            ok, msg, info, x_u8 = ok2, msg2, info2, x_u8_new

        if not ok:
            return _reject_payload(rid, model, msg, info or {})

        res = _infer_one(MODELS[model], x_u8, int(topk), rid)
        return _accept_payload(rid, model, info or {}, res)

    # B) JSON mode (base64) — supports single or batch with per-image request_id
    try:
        req = await request.json()

        model = req.get("model", "face")
        topk = int(req.get("topk", 3))
        imgs = req.get("images", None)
        if not imgs or not isinstance(imgs, list):
            raise HTTPException(400, "invalid request: 'images' must be a non-empty list of base64 strings")
        if model not in MODELS:
            raise HTTPException(400, f"model must be one of {list(MODELS.keys())}")

        req_ids = req.get("request_id")
        if isinstance(req_ids, list):
            rid_list = [str(x) for x in req_ids]
            if len(rid_list) != len(imgs):
                raise HTTPException(400, "invalid request: length of 'request_id' list must match length of 'images'")
        else:
            rid_single = _require_request_id(request, req_ids)
            rid_list = None

        def _process_one(img_b64: str, rid: str) -> Dict[str, Any]:
            img_bytes = base64.b64decode(img_b64)
            x_u8 = _pil_bytes_to_tensor(img_bytes)

            ok, msg, info = _run_gate(model, x_u8)
            ok2, msg2, info2, x_u8_new, fixed = _maybe_autofix_and_retry_gate(model, x_u8, ok, info or {})
            if fixed:
                ok, msg, info, x_u8 = ok2, msg2, info2, x_u8_new

            if not ok:
                return _reject_payload(rid, model, msg, info or {})

            res = _infer_one(MODELS[model], x_u8, topk, rid)
            return _accept_payload(rid, model, info or {}, res)

        if len(imgs) == 1:
            rid = rid_single if rid_list is None else rid_list[0]
            return _process_one(imgs[0], rid)

        # Batch: require per-image request_id list
        if rid_list is None:
            raise HTTPException(400, "invalid request: for batch 'images', 'request_id' must be a same-length list")

        items = []
        for img_b64, rid in zip(imgs, rid_list):
            items.append(_process_one(img_b64, rid))

        # Batch response is namespaced to avoid breaking single-image clients
        return {"model_used": model, "batch": items}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"invalid request: {e}")

# ---------------------- Health ----------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "device": str(DEVICE),
        # inference knobs
        "explore_on": EXPLORE_ON,
        "explore_mode": EXPLORE_MODE,
        "topn_pool": TOPN_POOL,
        "sample_T": SAMPLE_T,
        "uncertainty_metric": UNCERTAINTY_METRIC,
        "margin_th": MARGIN_TH,
        "entropy_th": ENTROPY_TH,
        "prob_cap": PROB_CAP,
        "min_prob": MIN_PROB,
        "tta": INFER_TTA,
        # gate runtime
        "scrfd_onnx": SCRFD_ONNX,
        "face_det_providers": FACE_DET_PROVIDERS,
    }
