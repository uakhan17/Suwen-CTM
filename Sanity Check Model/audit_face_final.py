#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============ hush optional fallback logs BEFORE any import ============
import os
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "1")
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # keep TF from touching your GPU

import sys, math, csv, json, argparse
from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD
from contextlib import contextmanager

# ===================== POLICY (choose one) =====================
# "strict"        – original spirit (harsh on small faces)
# "clinic_guided" – terminals with an on-screen frame; allow half-body if centered
# "clinic_wild"   – most permissive while still blocking ceilings; good for your dataset audit
POLICY = "clinic_wild"
# ===============================================================

# ----------------------------- Config -----------------------------
ONNX_DEFAULT = "/home/v3a/.insightface/models/scrfd_2.5g_bnkps.onnx"

INPUT_SIZES  = [(640,640), (960,960), (512,512), (320,320)]
DOWNSCALES   = [1100, 900, 700, 560, 420, 360, 320]
PADS         = [0.12, 0.25, 0.45, 0.60]

# clinic-friendly area window + center bias
if POLICY == "strict":
    AREA_MIN, AREA_MAX = 0.12, 0.98
    CENTER_TOL = 0.25
elif POLICY == "clinic_guided":
    AREA_MIN, AREA_MAX = 0.10, 0.98
    CENTER_TOL = 0.22
else:  # clinic_wild
    AREA_MIN, AREA_MAX = 0.08, 0.995
    CENTER_TOL = 0.22

NEAR_MISS_DELTA = 0.03
MAX_ROLL_DEG  = 20
MAX_YAW_ASYM  = 0.42      # profile gate via eye–nose asymmetry
NOSE_CX_MAX   = 0.32      # nose horizontal offset / box width
EPS = 1e-3
USE_MEDIAPIPE_FALLBACK_DEFAULT = True  # keeps ceilings out while rescuing odd close-ups

# =================== helpers: suppress C++ stderr ===================
@contextmanager
def suppress_cpp_stderr():
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield; return
    saved = os.dup(fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)

# =================== image transforms + mapping ===================
def _down_to_with_r(img, max_side):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side: return img, 1.0
    r = max_side / float(s)
    return cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA), r

def _pad_border_with_vals(img, pad=0.12, color=(114,114,114)):
    h, w = img.shape[:2]
    ph, pw = int(h*pad), int(w*pad)
    return cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=color), pw, ph

def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)

def _gamma(img, g=1.2):
    table = np.array([(i/255.0)**(1.0/g)*255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def _map_to_orig(box, kps, M, w0, h0):
    x1,y1,x2,y2,conf = box
    kps = kps.copy()
    t = M.get("type","identity")
    if t == "scale":
        r = M["r"];  x1/=r; y1/=r; x2/=r; y2/=r; kps/=r
    elif t == "pad":
        pw, ph = M["pw"], M["ph"]
        x1 -= pw; x2 -= pw; y1 -= ph; y2 -= ph
        kps[...,0] -= pw; kps[...,1] -= ph
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w0, int(x2)); y2 = min(h0, int(y2))
    return (x1, y1, x2, y2, float(conf)), kps

# =================== pose + geometry ===================
def yaw_asymmetry(kps):
    re, le, nose = np.asarray(kps[0]), np.asarray(kps[1]), np.asarray(kps[2])
    dr = float(np.linalg.norm(re - nose)); dl = float(np.linalg.norm(le - nose))
    return abs(dr - dl) / max(dr + dl, 1e-6)

def roll_deg_from_eyes(kps):
    e0, e1 = np.asarray(kps[0]), np.asarray(kps[1])
    left, right = (e1, e0) if e1[0] < e0[0] else (e0, e1)
    v = left - right
    ang = abs(math.degrees(math.atan2(v[1], v[0])))
    return ang if ang <= 90 else 180 - ang

def landmark_plausible_expr_tolerant(bbox, kps, area_frac, mode="normal"):
    x1,y1,x2,y2,_ = bbox
    W = max(1.0, x2-x1)

    # inside-box (loose)
    tol = 0.03 * W
    for (x,y) in kps:
        if not (x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol):
            return False, "关键点越界", {}

    # eyes above nose
    eyes_y = (kps[0][1] + kps[1][1]) * 0.5
    if not (eyes_y < kps[2][1]):
        return False, "眼鼻顺序异常", {}

    # eye distance vs width —— baby/macro tolerant
    eye_dist = float(np.hypot(kps[1][0]-kps[0][0], kps[1][1]-kps[0][1]))
    r_eye_w  = eye_dist / W
    if   mode == "baby":  low_bound = 0.14
    elif area_frac < 0.15: low_bound = 0.16
    elif area_frac < 0.20: low_bound = 0.20
    else:                  low_bound = 0.22
    if not (low_bound <= r_eye_w <= 0.88):
        return False, f"眼距比例异常({r_eye_w:.2f})", {}

    # nose near horizontal center (looser for macro)
    cx = (x1 + x2) * 0.5
    nose_off = abs(kps[2][0] - cx) / W
    nose_max = 0.35 if mode == "macro" else 0.28
    if nose_off > nose_max:
        return False, "鼻子偏离中心过大", {}

    # mouth width & vertical spacing —— relax for macro/tiny faces
    mouth_w   = float(np.hypot(kps[4][0]-kps[3][0], kps[4][1]-kps[3][1]))
    r_mouth_w = mouth_w / W if mouth_w > 0 else 0.0
    eyes_to_mouth = ((kps[3][1] + kps[4][1]) * 0.5 - eyes_y) / W
    upper = 0.85 if mode == "macro" else (0.78 if (area_frac >= 0.90 or area_frac < 0.15) else 0.72)

    if r_mouth_w < 0.22:  # pursed lips
        if not (eyes_y < (kps[3][1] + kps[4][1]) * 0.5): return False, "口部位置异常", {}
        if not (0.08 <= eyes_to_mouth <= upper):         return False, "上下比例异常(表情)", {}
        return True, "OK(表情异常：噘嘴)", {"pursed_lips": True}
    else:
        if not (0.12 <= eyes_to_mouth <= upper):         return False, f"上下比例异常({eyes_to_mouth:.2f})", {}
        return True, "OK", {}

# loosen small-but-centered; keep ceilings blocked
def adaptive_conf(area_frac, center_norm):
    """
    Area + center aware confidence.
    center_norm: 0 (center) .. 1 (corner)
    """
    # base by area
    if   area_frac >= 0.55: need = 0.50
    elif area_frac >= 0.40: need = 0.60
    elif area_frac >= 0.30: need = 0.70
    elif area_frac >= 0.20: need = 0.80
    elif area_frac >= 0.15: need = 0.85
    elif area_frac >= 0.12: need = 0.88
    elif area_frac >= 0.10: need = 0.90
    elif area_frac >= 0.08: need = 0.92
    else: return 1.01  # too small

    # center bonus for small faces (half-body)
    if area_frac < 0.20:
        if   center_norm <= 0.15: need -= 0.07
        elif center_norm <= 0.25: need -= 0.05
        elif center_norm <= 0.35: need -= 0.03

    return max(0.80, need)


# =================== detector loader & multi-try ===================
def load_scrfd(onnx_path: str, use_gpu: bool):
    det = SCRFD(model_file=onnx_path)
    providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    det.prepare(ctx_id=0, nms=0.4, providers=providers)
    return det, providers

def detect_face_robust(det, img_bgr):
    tries = []
    # originals
    for sz in INPUT_SIZES:
        tries.append((f"orig-{sz[0]}", img_bgr, sz, {"type":"identity"}))
    # downscales
    for ms in DOWNSCALES:
        im2, r = _down_to_with_r(img_bgr, ms)
        tries.append((f"down{ms}-640", im2, (640,640), {"type":"scale","r":r}))
    # pads
    for p in PADS:
        im2, pw, ph = _pad_border_with_vals(img_bgr, p)
        tries.append((f"pad{int(p*100)}-640", im2, (640,640), {"type":"pad","pw":pw,"ph":ph}))
    # contrast tweaks
    tries.append(("clahe-640", _clahe(img_bgr), (640,640), {"type":"identity"}))
    tries.append(("gamma12-640", _gamma(img_bgr, 1.2), (640,640), {"type":"identity"}))

    best = None
    for tag, im, sz, M in tries:
        if im is None: continue
        b, k = det.detect(im, input_size=sz)  # your SCRFD build uses only input_size
        if b is None or len(b)==0: 
            continue
        # keep best by max score for now; we will select primary later with center/area
        idx  = int(np.argmax(b[:,4])); conf = float(b[idx,4])
        if (best is None) or (conf > best[3]):
            best = (b, k, tag, conf, idx, M)
        if conf >= 0.90:
            break
    return best  # or None

# =================== mediapipe fallback (optional) ===================
def detect_fallback_mediapipe(img_bgr):
    with suppress_cpp_stderr():
        try:
            import mediapipe as mp
            from absl import logging as absl_logging
            absl_logging.set_verbosity(absl_logging.ERROR)
        except Exception:
            return None
        h, w = img_bgr.shape[:2]
        with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as det:
            res = det.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return None
        # collect all boxes to allow primary selection
        b_list, k_list = [], []
        for d in res.detections:
            rb = d.location_data.relative_bounding_box
            x1 = int(rb.xmin * w); y1 = int(rb.ymin * h)
            bw = int(rb.width * w); bh = int(rb.height * h)
            x2, y2 = x1 + bw, y1 + bh
            conf = float(d.score[0])
            rk = d.location_data.relative_keypoints  # RIGHT_EYE, LEFT_EYE, NOSE_TIP, MOUTH_CENTER, RIGHT_EAR, LEFT_EAR
            kps = np.array([
                [rk[0].x * w, rk[0].y * h],
                [rk[1].x * w, rk[1].y * h],
                [rk[2].x * w, rk[2].y * h],
                [rk[3].x * w, rk[3].y * h],
                [rk[3].x * w, rk[3].y * h],  # duplicate mouth center
            ], dtype=np.float32)
            b_list.append([x1,y1,x2,y2,conf]); k_list.append(kps)
        if not b_list:
            return None
        b = np.array(b_list, dtype=np.float32)
        k = np.stack(k_list, axis=0)
        return (b, k, "fallback-mediapipe", float(b[np.argmax(b[:,4]),4]), int(np.argmax(b[:,4])), {"type":"identity"})

# =================== primary selection & gating ===================
def select_primary(bboxes, kpss, M, w0, h0):
    """Choose ONE face to evaluate (largest + centered + confident)."""
    scores = []
    for j in range(bboxes.shape[0]):
        box_m, kps_m = _map_to_orig(bboxes[j], kpss[j], M, w0, h0)
        x1,y1,x2,y2,conf = box_m
        area = ((x2-x1)*(y2-y1))/float(w0*h0)
        cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
        dx = abs(cx - w0*0.5)/(w0*0.5); dy = abs(cy - h0*0.5)/(h0*0.5)
        center_norm = min(1.0, math.hypot(dx, dy))
        # prominence score: conf * center_weight * sqrt(area)
        score = conf * (0.6 + 0.4*(1.0 - center_norm)) * (max(area, 1e-6)**0.5)
        scores.append((score, j, box_m, kps_m, center_norm, area))
    scores.sort(reverse=True, key=lambda t: t[0])
    return scores[0]  # (score, idx, box_mapped, kps_mapped, center_norm, area_frac)

def find_secondary_faces(bboxes, kpss, M, w0, h0,
                         area_min=0.06, conf_min=0.60, center_max=0.70, primary_bbox=None):
    secs = []
    for j in range(bboxes.shape[0]):
        box_m, _ = _map_to_orig(bboxes[j], kpss[j], M, w0, h0)
        x1,y1,x2,y2,conf = box_m
        area = ((x2-x1)*(y2-y1))/float(w0*h0)
        cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
        dx = abs(cx - w0*0.5)/(w0*0.5); dy = abs(cy - h0*0.5)/(h0*0.5)
        center_norm = min(1.0, math.hypot(dx, dy))
        if (area >= area_min) and (conf >= conf_min) and (center_norm <= center_max):
            secs.append([x1,y1,x2,y2])
    # drop the primary if provided (IoU > 0.6)
    if primary_bbox and secs:
        ax1,ay1,ax2,ay2 = primary_bbox
        def iou(b):
            bx1,by1,bx2,by2 = b
            iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
            inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
            return inter/max(ua,1e-6)
        secs = [b for b in secs if iou(b) < 0.6]
    return secs

def gate_face_one(det, img_bgr, use_fallback=True):
    """
    Returns a dict with at least: ok(bool), reason(str).
    Depends on helpers/constants already in your file:
    - detect_face_robust, detect_fallback_mediapipe, _map_to_orig
    - yaw_asymmetry, roll_deg_from_eyes, landmark_plausible_expr_tolerant
    - adaptive_conf
    - AREA_MIN, AREA_MAX, MAX_YAW_ASYM, NOSE_CX_MAX, MAX_ROLL_DEG
    - EPS (default 1e-3), NEAR_MISS_DELTA (default 0.03)
    """
    import math, numpy as np

    h0, w0 = img_bgr.shape[:2]
    found = detect_face_robust(det, img_bgr)
    if found is None and use_fallback:
        found = detect_fallback_mediapipe(img_bgr)
    if found is None:
        return dict(ok=False, reason="未检测到人脸（多策略+回退仍失败）")

    bboxes, kpss, which_try, _, _, M = found
    num_faces = int(bboxes.shape[0])

    # ----- choose primary face (prominence = conf * center_weight * sqrt(area)) -----
    best_score, primary, mapped = -1.0, None, []
    for j in range(num_faces):
        box_m, kps_m = _map_to_orig(bboxes[j], kpss[j], M, w0, h0)
        x1,y1,x2,y2,conf = box_m
        area_frac = ((x2-x1)*(y2-y1))/float(w0*h0)
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        dx = abs(cx - w0*0.5)/(w0*0.5); dy = abs(cy - h0*0.5)/(h0*0.5)
        center_norm = min(1.0, math.hypot(dx, dy))
        score = float(conf) * (0.6 + 0.4*(1.0 - center_norm)) * (max(area_frac,1e-6)**0.5)
        mapped.append((box_m, kps_m, center_norm, area_frac))
        if score > best_score:
            best_score, primary = score, (box_m, kps_m, center_norm, area_frac)

    (box_m, kps_m, center_norm, area_frac) = primary
    x1,y1,x2,y2,conf = box_m
    
    if num_faces > 1:
        def iou(a,b):
            ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
            iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
            inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
            return inter/max(ua,1e-6)
        extras=[]
        for (b_m, k_m, c_norm, secondary_area_frac) in mapped:
            bb=[int(b_m[0]),int(b_m[1]),int(b_m[2]),int(b_m[3])]
            if iou(bb,[x1,y1,x2,y2])<0.6:
                print(f"  - Secondary Face Check: Area={secondary_area_frac:.4f} (needs >=0.015), Conf={float(b_m[4]):.4f} (needs >=0.50)")
                area=((bb[2]-bb[0])*(bb[3]-bb[1]))/float(w0*h0)
                if area>=0.01 and float(b_m[4])>=0.50:
                    extras.append(dict(bbox=bb, conf=float(b_m[4]), area=float(area), center=float(c_norm)))
        if extras:
            return dict(ok=False, reason=f"画面中检测到多个人脸，请仅保留患者入镜（{len(extras)}）",
                        which_try=which_try, num_faces=num_faces, conf=conf, bbox=[x1,y1,x2,y2],
                        area_frac=area_frac, extra_faces=extras)

    # ----- area bounds (allow macros up to ~1.0 if you set AREA_MAX accordingly) -----
    if not (AREA_MIN <= area_frac <= AREA_MAX):
        return dict(ok=False, reason=f"人脸离镜头过远/过近（{which_try}, area={area_frac:.2f}）",
                    which_try=which_try, num_faces=num_faces, conf=conf, bbox=[x1,y1,x2,y2], area_frac=area_frac)

    # ----- explicit multi-face rejection with clear reason -----
    

    # ----- decide mode for downstream tolerance -----
    if area_frac >= 0.85:                       mode = "macro"   # extreme close-up
    elif area_frac >= 0.50:                     mode = "large"   # big close-up
    elif area_frac < 0.13 and center_norm <= 0.30: mode = "baby"
    elif area_frac < 0.20 and center_norm <= 0.35: mode = "half"
    else:                                       mode = "normal"

    # ----- pose gates: yaw first, then roll -----
    yaw_a = yaw_asymmetry(kps_m)
    nose_off = abs(kps_m[2][0] - (x1+x2)*0.5) / max(1.0, (x2-x1))
    if (yaw_a > MAX_YAW_ASYM) or (nose_off > NOSE_CX_MAX):
        return dict(ok=False, reason=f"侧脸/转头过大（yaw={yaw_a:.2f}，nose_off={nose_off:.2f}）",
                    which_try=which_try, num_faces=num_faces, conf=conf, bbox=[x1,y1,x2,y2],
                    area_frac=area_frac, yaw_asym=yaw_a)

    roll = roll_deg_from_eyes(kps_m)
    if roll > MAX_ROLL_DEG:
        return dict(ok=False, reason=f"请保持手机水平（{which_try}, roll={roll:.1f}°）",
                    which_try=which_try, num_faces=num_faces, conf=conf, bbox=[x1,y1,x2,y2],
                    area_frac=area_frac, roll_deg=roll)

    # ----- geometry (expression-tolerant; handles macro & babies) -----
    ok_geo, geo_msg, flags = landmark_plausible_expr_tolerant(box_m, kps_m, area_frac, mode=mode)
    if not ok_geo:
        return dict(ok=False, reason=f"人脸几何不可信（{which_try}，{geo_msg}）",
                    which_try=which_try, num_faces=num_faces, conf=conf, bbox=[x1,y1,x2,y2], area_frac=area_frac)

    # ----- confidence need (area+center) + mode clamps + near-miss override -----
    need = adaptive_conf(area_frac, center_norm)
    if   mode == "large": need = min(need, 0.75)
    elif mode == "macro": need = min(need, 0.50)
    elif mode == "half":  need = min(need, 0.83)
    elif mode == "baby":  need = min(need, 0.80)

    EPS  = globals().get("EPS", 1e-3)
    NEAR = globals().get("NEAR_MISS_DELTA", 0.03)

    if float(conf) + EPS < need:
        if float(conf) + NEAR >= need:  # pass near-miss if geometry already OK
            return dict(ok=True, reason="OK(关键点通过/近阈值)",
                        which_try=which_try, num_faces=num_faces, conf=float(conf),
                        bbox=[x1,y1,x2,y2], area_frac=area_frac,
                        roll_deg=roll, yaw_asym=yaw_a,
                        landmarks=[float(v) for xy in kps_m for v in xy])
        return dict(ok=False, reason=f"人脸置信度不足（{which_try}, conf={float(conf):.2f} < {need:.2f} @area={area_frac:.2f}, center={center_norm:.2f}）",
                    which_try=which_try, num_faces=num_faces, conf=float(conf),
                    bbox=[x1,y1,x2,y2], area_frac=area_frac)

    # ----- success -----
    msg = "OK(表情异常)" if flags.get("pursed_lips") else "OK"
    return dict(ok=True, reason=msg,
                which_try=which_try, num_faces=num_faces, conf=float(conf),
                bbox=[x1,y1,x2,y2], area_frac=area_frac, roll_deg=roll, yaw_asym=yaw_a,
                landmarks=[float(v) for xy in kps_m for v in xy])


# =================== CSV audit over a folder ===================
def draw_viz(img_bgr, bbox, kps, text, ok_color=(0,255,0), bad_color=(0,0,255)):
    color = ok_color if text.startswith("OK") else bad_color
    x1,y1,x2,y2 = bbox
    cv2.rectangle(img_bgr, (x1,y1), (x2,y2), color, 2)
    for (x,y) in kps:
        cv2.circle(img_bgr, (int(x),int(y)), 2, color, -1)
    cv2.putText(img_bgr, text, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img_bgr

def audit_one(det, p: Path, use_fallback=True):
    rec = {"path": str(p)}
    img = cv2.imread(str(p))
    if img is None:
        rec.update(ok=False, reason="读取失败/非图像")
        return rec, None

    res = gate_face_one(det, img, use_fallback=use_fallback)
    rec.update(res)

    viz = None
    if res.get("bbox") and res.get("landmarks"):
        kps = np.array(res["landmarks"], dtype=np.float32).reshape(5,2)
        viz = draw_viz(img.copy(), res["bbox"], kps,
                       f"{res['reason']} | conf={res.get('conf',0):.2f} | area={res.get('area_frac',0):.2f} | yaw={res.get('yaw_asym',0):.2f} | roll={res.get('roll_deg',0):.1f}")
    return rec, viz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图片目录（递归）")
    ap.add_argument("--onnx", default=ONNX_DEFAULT, help="SCRFD onnx 路径 (2.5G KPS)")
    ap.add_argument("--out", default="face_detect_log.csv", help="输出CSV路径")
    ap.add_argument("--vizdir", default="", help="可选：保存可视化结果的目录")
    ap.add_argument("--use_gpu", action="store_true", help="若安装了 onnxruntime-gpu 则启用GPU")
    ap.add_argument("--no_fallback", action="store_true", help="禁用 mediapipe 回退")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.tif,.tiff", help="逗号分隔扩展名")
    args = ap.parse_args()

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        print(f"目录不存在: {img_dir}", file=sys.stderr); sys.exit(2)

    det, providers = load_scrfd(args.onnx, use_gpu=args.use_gpu)
    print(f"Providers: {providers}")

    vizdir = Path(args.vizdir) if args.vizdir else None
    if vizdir: vizdir.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip().lower() for e in args.exts.split(","))
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()

    headers = ["path","ok","reason","which_try","num_faces","conf","bbox","area_frac","roll_deg","yaw_asym","landmarks"]
    with open(args.out, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=headers)
        w.writeheader()
        for i, p in enumerate(files, 1):
            rec, viz = audit_one(det, p, use_fallback=not args.no_fallback)
            row = {k: rec.get(k, "") for k in headers}
            for k in ["bbox","landmarks"]:
                row[k] = json.dumps(row[k], ensure_ascii=False) if row[k] else ""
            w.writerow(row)
            if vizdir and viz is not None:
                cv2.imwrite(str(vizdir / (p.stem + "_viz.jpg")), viz)
            if i % 50 == 0:
                print(f"[{i}/{len(files)}] last: ok={rec.get('ok')} reason={rec.get('reason')}")

    print(f"Done. CSV: {args.out}")
    if vizdir: print(f"Viz saved to: {vizdir}")

if __name__ == "__main__":
    main()
