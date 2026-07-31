#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- hush logs BEFORE any heavy imports ----
import os
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtderr", "1")

import cv2, math, csv, json, argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

# ========= CONFIG (tune later if needed) =========
SCRFD_ONNX = "/home/v3a/.insightface/models/scrfd_2.5g_bnkps.onnx"
SCRFD_INPUT = (640, 640)

MAR_MIN = 0.20            # mouth open threshold (inner-lip gap / mouth width)
MAR_TONGUE = 0.28         # if MAR>=this, allow lower color fraction
TONGUE_FRAC_MIN = 0.28    # fraction of tongue-like pixels in mouth ROI
TONGUE_FRAC_RELAX = 0.20  # relaxation when MAR is big

MAR_LOWER = 0.10            # allow this low if protrusion is strong
PROTRUDE_NORM_MIN = 0.16    # tip depth measured along mouth normal, / mouth_width
MASK_FRAC_STRONG = 0.35     # strong tongue-color coverage to allow low MAR

# white-coated tongue tuning (if not already in your file)
WHITE_MIN_L = 155
WHITE_A_LO, WHITE_A_HI = 115, 165
WHITE_B_LO, WHITE_B_HI = 112, 175
HIGHLIGHT_L = 235
WHITE_FRAC_MIN = 0.18       # relax target if clear white coat

# central blob fallback (area lowered but shape stricter)
CENTRAL_MIN_AREA = 0.06
CENTRAL_MIN_SOLIDITY = 0.70

ROLL_MAX = 25.0           # deg
AREA_MIN_FACE = 0.06      # face-box area floor (as frac of image)
CONF_NEED = 0.75          # face conf floor if we rely on SCRFD
ROI_PAD = 0.25            # expand the outer-lip bbox when cropping ROI
# ================================================
def protrusion_from_corners(roi_shape, left_xy, right_xy, mask_bin, roi_xy):
    """
    Approximate tongue protrusion depth measured from the lip-corner baseline.
    - roi_shape: (H, W) of the ROI
    - left_xy, right_xy: mouth corner coords in full-image space
    - mask_bin: uint8 mask within ROI (1 for tongue-like)
    - roi_xy: (rx1, ry1) top-left of ROI in full-image space
    Returns: tip_depth_norm (relative to mouth width), below_frac (mask fraction below baseline)
    """
    H, W = roi_shape[:2]
    rx1, ry1 = roi_xy

    # baseline y at the midpoint between corners
    baseline_y = (left_xy[1] + right_xy[1]) * 0.5 - ry1
    mouth_w = float(np.hypot(right_xy[0]-left_xy[0], right_xy[1]-left_xy[1]) + 1e-6)

    ys, xs = np.where(mask_bin > 0)
    if ys.size == 0:
        return 0.0, 0.0

    # depth: how far the mask extends below the baseline (downwards is +y)
    depth = float(np.max(ys - baseline_y))  # could be negative -> use max
    tip_depth_norm = max(0.0, depth) / mouth_w

    # fraction of mask pixels clearly below baseline (with a tiny margin)
    below = ys > (baseline_y + 0.03 * mouth_w)
    below_frac = float(np.mean(below)) if ys.size > 0 else 0.0
    return tip_depth_norm, below_frac

def load_scrfd(onnx_path=SCRFD_ONNX, use_gpu=True):
    from insightface.model_zoo.scrfd import SCRFD
    det = SCRFD(model_file=onnx_path)
    providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    det.prepare(ctx_id=0, nms=0.4, providers=providers)
    return det, providers

DET, PROVIDERS = load_scrfd(use_gpu=True)

# ---------- simple pose helpers from SCRFD 5-point kps ----------
def yaw_asymmetry(kps):
    re, le, nose = kps[0], kps[1], kps[2]
    dr = np.linalg.norm(re - nose); dl = np.linalg.norm(le - nose)
    return float(abs(dr - dl) / max(dr + dl, 1e-6))

def roll_deg_from_eyes(kps):
    e0, e1 = kps[0], kps[1]
    left, right = (e1, e0) if e1[0] < e0[0] else (e0, e1)
    v = left - right
    ang = abs(math.degrees(math.atan2(v[1], v[0])))
    return float(ang if ang <= 90 else 180 - ang)

# ---------- MediaPipe FaceMesh lips ----------
def mp_lips_landmarks(img_bgr):
    try:
        import mediapipe as mp
    except Exception:
        return None
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w = img_bgr.shape[:2]
    idx_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idx_outer], dtype=np.float32)
    left = np.array([lm[61].x * w, lm[61].y * h], dtype=np.float32)
    right = np.array([lm[291].x * w, lm[291].y * h], dtype=np.float32)
    up_in = np.array([lm[13].x * w, lm[13].y * h], dtype=np.float32)
    lo_in = np.array([lm[14].x * w, lm[14].y * h], dtype=np.float32)
    return dict(outer_pts=pts, left=left, right=right, up_in=up_in, lo_in=lo_in)

# ---------- color mask in Lab: pick warm/pink-ish cluster ----------
def tongue_mask_lab(roi_bgr, return_debug=False):
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # suppress tiny specular highlights
    spec = (L >= HIGHLIGHT_L)
    if spec.any():
        L = L.copy()
        L[spec] = np.median(L[~spec]) if (~spec).any() else 220

    # branch A: classic pink/red tongue
    mask_red = (a > 135) & (a < 185) & (b > 120) & (b < 180) & (L > 40) & (L < 240)

    # branch B: white/gray coat tongue
    mask_white = (L > WHITE_MIN_L) & \
                 (a >= WHITE_A_LO) & (a <= WHITE_A_HI) & \
                 (b >= WHITE_B_LO) & (b <= WHITE_B_HI)

    # ### --- MODIFICATION START --- ###
    # branch C: purple/violet tongue (reddish 'a', bluish 'b')
    mask_purple = (a > 140) & (a < 190) & (b > 90) & (b < 120) & (L > 40)
    # branch D: light-yellow tongue (yellowish 'b', neutral 'a')
    mask_yellow = (a > 110) & (a < 140) & (b > 145) & (b < 195) & (L > 100)
    # ### --- MODIFICATION END --- ###


    # soft k-means on (a,b) to catch warm-ish cluster if present
    ab = np.float32(np.stack([a.flatten(), b.flatten()], axis=1))
    K = 2
    _, labels, centers = cv2.kmeans(
        ab, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0),
        2, cv2.KMEANS_PP_CENTERS
    )
    warm_idx = int(np.argmax(centers[:, 0]))  # higher 'a' = redder
    kmm = labels.reshape(a.shape) == warm_idx

    # ### --- MODIFICATION START --- ###
    # Combine all masks together
    mask = (mask_red | mask_white | kmm | mask_purple | mask_yellow).astype(np.uint8) * 255
    # ### --- MODIFICATION END --- ###

    # clean up
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    out = (mask > 0).astype(np.uint8)

    if return_debug:
        # Note: frac_red and frac_white are now less meaningful as standalone debug values
        return out, float(mask_red.mean()), float(mask_white.mean())
    return out

# ---------- fallback when no lips/no face ----------
def central_reddish_blob(img_bgr):
    h, w = img_bgr.shape[:2]
    L, a, b = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))

    # ### --- MODIFICATION START --- ###
    # Expanded color check for TCM tongues
    red = ((a > 130) & (b > 115) & (L > 25) & (L < 245))
    white = ((L > WHITE_MIN_L) &
             (a >= WHITE_A_LO) & (a <= WHITE_A_HI) &
             (b >= WHITE_B_LO) & (b <= WHITE_B_HI))
    purple = (a > 140) & (a < 190) & (b > 90) & (b < 120) & (L > 40)
    yellow = (a > 110) & (a < 140) & (b > 145) & (b < 195) & (L > 100)
    mask = (red | white | purple | yellow).astype(np.uint8) * 255
    # ### --- MODIFICATION END --- ###
    
    mask = cv2.medianBlur(mask, 5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return False
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt) / (w * h + 1e-6)
    if area < CENTRAL_MIN_AREA:  # lowered from 0.10 to 0.06
        return False
    hull = cv2.convexHull(cnt)
    solidity = cv2.contourArea(cnt) / (cv2.contourArea(hull) + 1e-6)
    if solidity < CENTRAL_MIN_SOLIDITY:
        return False
    M = cv2.moments(cnt)
    if M["m00"] == 0: return False
    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    center = math.hypot((cx - w*0.5)/(w*0.5), (cy - h*0.5)/(h*0.5))
    return center <= 0.35

# ---------- the gate itself (works for mouth-only photos) ----------
def gate_tongue(img_bgr) -> Tuple[bool, str, Dict[str, Any]]:
    h, w = img_bgr.shape[:2]

    # 1) FaceMesh lips first
    lips = mp_lips_landmarks(img_bgr)
    if lips is not None:
        # FaceMesh branch ...
        left, right, up_in, lo_in = lips["left"], lips["right"], lips["up_in"], lips["lo_in"]
        mouth_w = float(np.linalg.norm(right - left) + 1e-6)
        mar = float(np.linalg.norm(lo_in - up_in) / mouth_w)

        # build ROI from outer lips
        pts = lips["outer_pts"]
        x_min,y_min = np.min(pts, axis=0); x_max,y_max = np.max(pts, axis=0)
        pad_x = ROI_PAD * (x_max - x_min); pad_y = ROI_PAD * (y_max - y_min)
        rx1 = max(0, int(x_min - pad_x)); ry1 = max(0, int(y_min - pad_y))
        rx2 = min(w, int(x_max + pad_x)); ry2 = min(h, int(y_max + pad_y))
        roi = img_bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return False, "口部区域无效，请重拍", {"stage":"facemesh"}

        # tongue-like mask (red ∪ white)
        mask, frac_red, frac_white = tongue_mask_lab(roi, return_debug=True)
        frac = float(mask.mean())
        need_frac = TONGUE_FRAC_RELAX if mar >= MAR_TONGUE else TONGUE_FRAC_MIN
        if frac_white >= WHITE_FRAC_MIN:
            need_frac = min(need_frac, WHITE_FRAC_MIN)

        # NEW: tongue protrusion metric relative to the lip-corner baseline
        tip_norm, below_frac = protrusion_from_corners(
            roi.shape, left, right, (mask>0).astype(np.uint8), (rx1, ry1)
        )

        # Accept rules:
        # A) normal: MAR>=MAR_MIN and enough tongue fraction
        ok_normal = (mar >= MAR_MIN) and (frac >= need_frac)

        # B) narrow-slit override: very small MAR but clear protrusion & coverage
        ok_slit = (mar >= MAR_LOWER) and (
            (tip_norm >= PROTRUDE_NORM_MIN and frac >= (need_frac - 0.06)) or
            (frac >= MASK_FRAC_STRONG and below_frac >= 0.40)
        )

        if not (ok_normal or ok_slit):
            return False, f"请张大嘴并伸出舌头（张口不足/舌体不明显，mar={mar:.3f}, tip={tip_norm:.3f}, frac={frac:.2f})", {
                "stage":"facemesh", "mar":mar, "tongue_frac":frac, "frac_red":frac_red,
                "frac_white":frac_white, "tip_norm":tip_norm, "below_frac":below_frac,
                "roi":[rx1,ry1,rx2,ry2]
            }

        # shape sanity to reject thin lip lines
        cnts, _ = cv2.findContours((mask*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False, "舌体区域过小，请靠近镜头并伸出舌头", {"stage":"facemesh", "mar":mar}
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt); sol = cv2.contourArea(cnt)/(cv2.contourArea(hull)+1e-6)
        if sol < 0.58:  # slightly relaxed for real tongues with fissures
            return False, "舌体形状不稳定，请重新伸舌拍摄", {"stage":"facemesh", "solidity":sol, "mar":mar}

        return True, "OK", {"stage":"facemesh", "mar":mar, "tongue_frac":frac,
                            "frac_red":frac_red, "frac_white":frac_white,
                            "tip_norm":tip_norm, "below_frac":below_frac,
                            "solidity":sol, "roi":[rx1,ry1,rx2,ry2]}


    # 2) No lips -> try SCRFD to estimate lower-face ROI
    try:
        b, k = DET.detect(img_bgr, input_size=SCRFD_INPUT)
    except Exception:
        b = None
    if b is not None and len(b) > 0:
        idx = int(np.argmax(b[:,4])); x1,y1,x2,y2,conf = b[idx]
        x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2); conf=float(conf)
        area = ((x2-x1)*(y2-y1))/(w*h+1e-6)
        if area >= AREA_MIN_FACE and conf >= (CONF_NEED - 0.05):
            fw, fh = x2 - x1, y2 - y1
            mx1 = max(0, int(x1 - 0.05*fw)); mx2 = min(w, int(x2 + 0.05*fw))
            my1 = max(0, int(y1 + 0.45*fh)); my2 = min(h, int(y2))
            roi = img_bgr[my1:my2, mx1:mx2]
            if roi.size > 0:
                mask = tongue_mask_lab(roi)
                frac = float(mask.mean())
                if frac >= max(TONGUE_FRAC_MIN, 0.30):
                    cnts,_ = cv2.findContours((mask*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        hull = cv2.convexHull(cnt)
                        sol = cv2.contourArea(cnt)/(cv2.contourArea(hull)+1e-6)
                        if sol >= 0.55:
                            return True, "OK", {"stage":"scrfd_lower_face", "tongue_frac":frac,
                                                "solidity":sol, "roi":[mx1,my1,mx2,my2]}

    # 3) Final fallback
    if central_reddish_blob(img_bgr):
        return True, "OK(中心大面积舌体样区域)", {"stage":"central_blob"}

    return False, "未检测到舌体，请正对镜头、张大嘴并伸出舌头重拍", {"stage":"final_fail"}

# ---------- viz helper ----------
def draw_viz(img_bgr, info, ok, msg):
    out = img_bgr.copy()
    roi = info.get("roi")
    if roi:
        x1,y1,x2,y2 = roi
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0) if ok else (0,0,255), 2)
    cv2.putText(out, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,255,0) if ok else (0,0,255), 2, cv2.LINE_AA)
    return out

# ---------- batch audit over a folder ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图片目录（递归）")
    ap.add_argument("--out", default="tongue_gate_log.csv", help="输出CSV路径")
    ap.add_argument("--vizdir", default="", help="可选：保存可视化图")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.tif,.tiff", help="扩展名（逗号分隔）")
    args = ap.parse_args()

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        raise SystemExit(f"目录不存在: {img_dir}")

    vizdir = Path(args.vizdir) if args.vizdir else None
    if vizdir: vizdir.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip().lower() for e in args.exts.split(","))
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()

    headers = ["path","ok","reason","stage","mar","tongue_frac","solidity","roi"]
    with open(args.out, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=headers)
        w.writeheader()
        for i, p in enumerate(files, 1):
            img = cv2.imread(str(p))
            if img is None:
                row = dict(path=str(p), ok=False, reason="读取失败/非图像")
                w.writerow(row); continue

            ok, msg, info = gate_tongue(img)
            row = dict(
                path=str(p),
                ok=ok,
                reason=msg,
                stage=info.get("stage",""),
                mar=f"{info.get('mar',''):.4f}" if "mar" in info else "",
                tongue_frac=f"{info.get('tongue_frac',''):.4f}" if "tongue_frac" in info else "",
                solidity=f"{info.get('solidity',''):.4f}" if "solidity" in info else "",
                roi=json.dumps(info.get("roi",""), ensure_ascii=False) if "roi" in info else ""
            )
            w.writerow(row)

            if vizdir:
                viz = draw_viz(img, info, ok, msg)
                cv2.imwrite(str(vizdir / f"{p.stem}_viz.jpg"), viz)

            if i % 50 == 0:
                print(f"[{i}/{len(files)}] last: ok={ok} reason={msg}")

    print(f"Done. CSV: {args.out}")
    if vizdir: print(f"Viz saved to: {vizdir}  |  Providers: {PROVIDERS}")

if __name__ == "__main__":
    main()