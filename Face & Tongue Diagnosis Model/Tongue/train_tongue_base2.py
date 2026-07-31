from __future__ import annotations
import argparse, os, pathlib, random, time
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import json
from datasets import (
    TongueDataset, FaceDataset,                  # legacy
    MultiViewTongueDataset, collate_multiview,   # new
    collate_multitask, ALL_LABELS
)
from collections import defaultdict
from model import MultiTaskNet
from tqdm import tqdm
from pathlib import Path
import numpy as np
from torch.optim import AdamW
# --------------------------- CLI ---------------------------
# VIEW_FOR_HEAD = {
#     '舌质_色'  : 'base',     # will get MixUp on the fly
#     '舌苔_苔质': 'allaug',
#     '舌质_形'  : 'cutmix',
#     '舌质_态'  : 'cutmix',
#     '舌苔_苔色': 'cutmix',
#     '舌质_神'  : 'base',
# }
# MIXUP_HEADS = {'舌质_色'}

LA_HEADS = {'舌质_色', '舌质_形', '舌苔_苔质'}
LA_TAU   = 1.0
CLASS_COUNTS = {
    '舌质_色'  : [710, 306, 127, 49, 1],
    '舌质_形'  : [98, 338, 144, 103, 345, 115, 1],
    '舌苔_苔质': [56, 643, 164, 126, 209, 9, 55, 37],
}

MIXUP_HEADS = {'None'}
task_names_ = ['舌质_色','舌苔_苔质','舌质_形','舌质_态','舌苔_苔色','舌质_神']
VIEW_FOR_HEAD = {t: 'base' for t in task_names_}

def build_logit_adjust(class_counts: dict[str, list[int]], tau: float, device: torch.device):
    """Return dict {task: 1D tensor [C]} with tau*log(prior) (clamped to avoid log(0))."""
    la = {}
    for t, cnt in class_counts.items():
        c = torch.tensor(cnt, dtype=torch.float32, device=device)
        c = c.clamp_min(1.0)                         # avoid log(0) for zero-count classes
        prior = c / c.sum()                          # log of prior; using raw log(c) also works
        la[t] = tau * torch.log(prior)
    return la

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--img_root', required=True)
    p.add_argument('--tasks', choices=['tongue', 'face'], default='tongue')
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--cv', type=int, default=0)

    # RandAug prob (applied in Dataset)
    p.add_argument('--ra_prob', type=float, default=0)

    # MixUp & CutMix
    p.add_argument('--mix_prob', type=float, default=0)
    p.add_argument('--cutmix_prob', type=float, default=0)
    p.add_argument('--beta', type=float, default=1.0)

    # extras
    p.add_argument('--gradnorm', action='store_true')
    p.add_argument('--clip', type=float, default=0.0)
    p.add_argument('--out_dir', default='checkpoints')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

# ---------------------- helpers ----------------------

def get_dataset(args):
    ds_kwargs = dict(ra_prob=args.ra_prob)
    if args.tasks == 'tongue':
        # use multi‑view wrapper, not the single‑view class
        return MultiViewTongueDataset(args.csv, args.img_root, **ds_kwargs)
    return FaceDataset(args.csv, args.img_root, **ds_kwargs)

###############################################################################
#                               SafeGradNorm                                  #
###############################################################################
def safe_gradnorm(loss_dict, shared_module, alpha=1.5, eps=1e-8):
    """
    GradNorm with epsilon guards.
    loss_dict: {task_name: scalar loss (tensor)}
    shared_module: backbone nn.Module
    """
    device = next(shared_module.parameters()).device

    # --- initialise persistent state ---
    if not hasattr(safe_gradnorm, "task_order"):
        safe_gradnorm.task_order = list(loss_dict.keys())
        safe_gradnorm.w = torch.ones(len(safe_gradnorm.task_order),
                                     device=device, requires_grad=True)
        safe_gradnorm.L0 = None

    task_order = safe_gradnorm.task_order
    w = safe_gradnorm.w

    # vector of current losses in fixed order
    L_vec = torch.stack([loss_dict[t] for t in task_order])

    if safe_gradnorm.L0 is None:
        # baseline magnitude (add eps to avoid zero division)
        safe_gradnorm.L0 = L_vec.detach() + eps

    L0 = safe_gradnorm.L0

    # ----- 1. weighted per‑task losses -----
    weighted_losses = [w[i] * L_vec[i] for i in range(len(task_order))]
    weighted_total = torch.stack(weighted_losses).sum()

    # ----- 2. compute gradient norms G_i of w_i * L_i -----
    shared_params = [p for p in shared_module.parameters() if p.requires_grad]
    G_list = []
    for wl in weighted_losses:
        grads = torch.autograd.grad(wl, shared_params,
                                    retain_graph=True, create_graph=True)
        # norm over all shared params
        g_norm = torch.stack([g.norm() for g in grads]).mean()
        G_list.append(g_norm + eps)
    G = torch.stack(G_list)
    G_avg = G.mean()

    # ----- 3. inverse training rates r_i -----
    inv_rate = (L_vec / L0)
    inv_rate = inv_rate / (inv_rate.mean() + eps)

    # target norms
    target = (inv_rate ** alpha) * G_avg

    # ----- 4. GradNorm loss to update w -----
    gradnorm_loss = torch.abs(G - target.detach()).sum()
    # update only w
    gradnorm_loss.backward(retain_graph=True)

    # ----- 5. renormalise weights (Σ w_i = T) -----
    with torch.no_grad():
        w.clamp_(min=0.)
        w.mul_(len(task_order) / (w.sum() + eps))

    # return weighted sum of original (unweighted) losses
    # (detach w so outer backward doesn't hit create_graph path twice)
    return torch.sum(w.detach() * L_vec)

def rand_bbox(W: int, H: int, lam: float):
    cut_rat = (1. - lam) ** .5
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def save_split(save_dir, fold_id, tr_idx, va_idx, base_ds):
    """
    将当前 fold 的索引和文件名保存到 disk
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    def idx2path(idxs):
        out = []
        for i in idxs:
            name = base_ds.fnames[i]
            p = base_ds.id2p.get(name)
            if p is None:
                stem = Path(name).stem
                p = base_ds.id2p.get(stem)
            if p is None:
                raise RuntimeError(f"Can't find image path for '{name}'")
            out.append(str(p))
        return out
    split_dict = {
        'train_idx': np.asarray(tr_idx, dtype=int).tolist(),
        'val_idx':   np.asarray(va_idx, dtype=int).tolist(),
        'train_files': idx2path(tr_idx),
        'val_files':   idx2path(va_idx)
    }

    with open(save_dir / f'fold_{fold_id}.json', 'w', encoding='utf-8') as f:
        json.dump(split_dict, f, ensure_ascii=False, indent=2)
# ---------------------- training loop ----------------------

def train_fold(tr_idx: List[int], va_idx: List[int], fold: str, args):
    # base_ds_ = get_dataset(args)
    # tr_ds, va_ds = Subset(base_ds, tr_idx), Subset(base_ds, va_idx)
    if args.tasks == 'tongue':
        base_tr = MultiViewTongueDataset(args.csv, args.img_root, is_train=True,  img_size=224, ra_prob=0.0)
        base_va = MultiViewTongueDataset(args.csv, args.img_root, is_train=False, img_size=224, ra_prob=0.0)
        tr_ds, va_ds = Subset(base_tr, tr_idx), Subset(base_va, va_idx)
        collate_fn = collate_multiview
        meta = base_tr
    else:
        base_ds = get_dataset(args)
        tr_ds, va_ds = Subset(base_ds, tr_idx), Subset(base_ds, va_idx)
        collate_fn = collate_multitask
        meta = base_tr
    
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_fn)

    nc = meta.nc
    task_names = list(nc)
    model = MultiTaskNet(nc, backbone=args.backbone).to(args.device)
    logit_adjust = build_logit_adjust(CLASS_COUNTS, LA_TAU, device=torch.device(args.device))

    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # head_params = [p for n,p in model.named_parameters() if n.startswith('heads.')]
    # bb_params   = [p for n,p in model.named_parameters() if not n.startswith('heads.')]
    # opt = AdamW([
    # {'params': bb_params,   'lr': 5e-4,  'weight_decay': 1e-4},
    # {'params': head_params, 'lr': 1.5e-3,'weight_decay': 1e-4},])
    base_lr = 5e-4
    mult = {
        '舌质_形': 3.0,
        '舌苔_苔色': 3.0,
        '舌质_色': 1.0,       # ↓ was hurt by high LR
        '舌苔_苔质': 1.0,     # ↓ was hurt by high LR
        '舌质_态': 3.0,
        '舌质_神': 3.0,
    }

    bb_params = [p for n,p in model.named_parameters() if not n.startswith('heads.')]
    head_groups = []
    for t, head in model.heads.items():
        head_groups.append({'params': head.parameters(), 'lr': base_lr * mult[t], 'weight_decay': 1e-4})

    opt = torch.optim.AdamW(
        [{'params': bb_params, 'lr': base_lr, 'weight_decay': 1e-4}, *head_groups]
    )
    # GradNorm init
    W = torch.ones(len(task_names), device=args.device, requires_grad=False)
    alpha, L0 = 1.5, None

    out_dir = pathlib.Path(args.out_dir)/f'fold{fold}'; out_dir.mkdir(parents=True, exist_ok=True)

    for ep in tqdm(range(1, args.epochs+1),colour='green'):
        model.train() 
        ep_loss=0 
        t0=time.time()
        iter_losses = []                                   # ← 新建列表
        pbar = tqdm(tr_loader, desc=f"Fold {fold} Ep {ep}", ncols=100)
        
        for step, (views, gts, fnames) in enumerate(pbar):
            for k in views:
                views[k] = views[k].to(args.device)
            gts = {k: v.to(args.device) for k, v in gts.items()}
            # ---------------- batch‑level CutMix (cutmix view only) ----
            if random.random() < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                idx = torch.randperm(views['cutmix'].size(0)).to(args.device)
                # CutMix on images
                x1,y1,x2,y2 = rand_bbox(views['cutmix'].size(3),views['cutmix'].size(2), lam)
                views['cutmix'][:,:,y1:y2,x1:x2] = views['cutmix'][idx,:,y1:y2,x1:x2]
                lam = 1 - ((x2-x1)*(y2-y1)/(views['cutmix'].size(-1)*views['cutmix'].size(-2)))
                tgt_b = {k: v[idx] for k, v in gts.items()}
            else:
                lam, tgt_b = 1.0, None
            # --------------- backbone passes (3 views) ------------------
            feat_base   = model.backbone(views['base'])
            feat_cutmix = model.backbone(views['cutmix'])
            feat_allaug = model.backbone(views['allaug'])
            # GAP if ConvNeXt returned 4‑D
            if feat_base.ndim == 4:
                feat_base = feat_base.mean(dim=[2, 3])
            if feat_cutmix.ndim == 4:
                feat_cutmix = feat_cutmix.mean(dim=[2, 3])
            if feat_allaug.ndim == 4:
                feat_allaug = feat_allaug.mean(dim=[2, 3])

            task_losses = {}
            for t in task_names:
                n_cls = nc[t]                           # 类别数
                # pick the feature view this head prefers
                view_name = VIEW_FOR_HEAD[t]
                feat = {'base': feat_base,
                        'cutmix': feat_cutmix,
                        'allaug': feat_allaug}[view_name]

                # -------- MixUp only for 舌质_色 -------------------------
                if t in MIXUP_HEADS and random.random() < args.mix_prob:
                    lam_mu = np.random.beta(args.beta, args.beta)
                    idx = torch.randperm(feat.size(0)).to(args.device)
                    feat = lam_mu*feat + (1-lam_mu)*feat[idx]
                    y_a = F.one_hot(gts[t], n_cls).float()
                    y_b = F.one_hot(gts[t][idx], n_cls).float()
                    target = lam_mu*y_a + (1-lam_mu)*y_b
                else:
                    y_a = F.one_hot(gts[t], n_cls).float()
                    if view_name == 'cutmix' and tgt_b is not None:
                        y_b = F.one_hot(tgt_b[t], n_cls).float()
                        target = lam*y_a + (1-lam)*y_b
                    else:
                        target = y_a

                logits = model.heads[t](feat)
                # ---- Balanced Softmax: add per-class log-prior to logits at training time ----
                if t in LA_HEADS:
                    logits = logits + logit_adjust[t].unsqueeze(0)   # broadcast over batch
                log_prob = F.log_softmax(logits, dim=-1)
                task_losses[t] = -(target * log_prob).sum(dim=-1).mean()
 
            # -------- GradNorm --------
            eps = 1e-8                         # <‑‑ single global constant is enough
            loss_vec = torch.stack([task_losses[t] for t in task_names])
            if args.gradnorm:
                if L0 is None:
                    # store baseline magnitudes once; add eps so zero‑loss heads are safe
                    L0 = loss_vec.detach() + eps

                # ------------------------------------------------------------------
                weighted = (W * loss_vec).sum()
                opt.zero_grad()
                weighted.backward(retain_graph=True)

                # 1) gradient norms for each head
                grad_norm = torch.stack([
                    model.heads[t].weight.grad.norm() + eps   # +eps so never zero
                    for t in task_names
                ])
                G_avg = grad_norm.mean()

                # 2) inverse training rate
                inv_rate = loss_vec.detach() / L0          # denominator ≥ eps
                inv_rate = inv_rate / (inv_rate.mean() + eps) # normalise, eps‑guard
                target   = (inv_rate ** alpha) * G_avg        # eq. (4) in GradNorm

                # 3) update weights
                with torch.no_grad():
                    W *= grad_norm / (target + eps)           # eq. (6) with guard
                    # replace NaN / Inf that might still slip through
                    W = torch.nan_to_num(W, nan=1.0, posinf=1.0, neginf=1.0)
                    W /= W.mean() + eps                       # re‑normalise
            else:
                weighted = loss_vec.sum()
                opt.zero_grad()
                weighted.backward()

            if args.clip>0: 
                clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            iter_loss = weighted.item()
            bs = next(iter(views.values())).size(0)
            ep_loss += iter_loss * bs
        # tqdm 进度条显示
            pbar.set_postfix(loss=f"{iter_loss:.4f}")
            iter_losses.append({
            "epoch": ep,
            "iter":  step,
            "loss":  iter_loss
        }) 
            # ep_loss += weighted.item()*imgs.size(0)
        df = pd.DataFrame(iter_losses)
        df.to_csv(out_dir / f"fold{fold}_ep{ep:03d}_iterloss.csv",index=False, encoding="utf-8")
        # ---------------- validation ----------------
        if ep%10 == 0:
            model.eval()
            val_loss = 0.0

            logit_task = {t: [] for t in task_names}   # 收集 raw logits
            truth_task = {t: [] for t in task_names}

            hi_preds: list[dict] = []                 # 每张图像的高置信结果

            with torch.no_grad():
                for views, gts, fnames in va_loader:   # collate 返回 (imgs, labels, filenames)
                    base_img = views['base'].to(args.device)
                    gts = {k: v.to(args.device) for k, v in gts.items()}
                    feat_base = model.backbone(base_img)

                    if feat_base.ndim == 4:
                        feat_base = feat_base.mean(dim=[2,3])
                    outs = {t: model.heads[t](feat_base) for t in task_names}                # dict{t: logits [B, C]}

                    # -------- 累计 val_loss 与指标 --------
                    for t in task_names:
                        val_loss += F.cross_entropy(outs[t], gts[t],reduction="sum").item()              # 累加总和
                        logit_task[t].append(outs[t].cpu())
                        truth_task[t].append(gts[t].cpu())

                    # -------- 逐图片高置信 --------
                    B = views['base'].size(0)
                    for i in range(B):
                        img_dict = {}
                        for t in task_names:
                            names = ALL_LABELS[t]                 # 中文标签
                            row   = outs[t][i].softmax(dim=0).cpu()  # 概率向量

                            # ① 先取 >0.5 的所有标签
                            pairs = [(names[j], float(row[j]))
                                     for j in range(len(row)) if row[j] > 0.5]

                            # ② 若都 ≤0.5，则回退到 softmax Top-3
                            if not pairs:
                                topk = row.topk(k=min(3, len(row))).indices
                                pairs = [(names[j], float(row[j])) for j in topk]

                            pairs.sort(key=lambda x: -x[1])       # 概率降序
                            img_dict[t] = pairs
                        hi_preds.append({"file": fnames[i], "pred": img_dict})

            # -------- 平均 val_loss --------
            val_loss /= len(va_ds)

            # -------- 任务级指标 --------
            metrics = {}
            top_k = 3             # 这里可以改成任意 k

            for t in task_names:
                logits = torch.cat(logit_task[t], dim=0)   # [N, C]
                y_true = torch.cat(truth_task[t], dim=0)   # [N]

                # Top-1 (普通) 结果
                y_pred = logits.argmax(dim=1)

                acc_top1 = (y_pred == y_true).float().mean().item()
                f1_top1  = f1_score(y_true.numpy(),y_pred.numpy(),average='macro',zero_division=0)

                # ---------- Top-k ----------
                # 取每行最高 k 个类别索引:  topk_idx shape = [N, k]
                C = logits.size(1)          # 当前任务的类别数
                k_eff = min(top_k, C)       # 若 C<k，则退化为 C
                if k_eff < top_k:
                    acc_topk = None          # 二分类时 Top-3 恒为 1，也可设为 None
                else:
                    topk_idx = logits.topk(k=k_eff, dim=1).indices
                    correct_topk = (topk_idx == y_true.unsqueeze(1)).any(dim=1)
                    acc_topk = correct_topk.float().mean().item()

                metrics[t] = {
                    'ACC@1':  acc_top1,
                    'F1@1':   f1_top1,
                    f'ACC@{top_k}': acc_topk
                }

            print(f"Fold{fold} Ep{ep} train={ep_loss/len(tr_ds):.4f} val={val_loss:.4f}")
            for t, m in metrics.items():
                acc3 = m[f'ACC@{top_k}']
                acc3_str = f"{acc3:.3f}" if acc3 is not None else "—"
                print(f"  {t}: ACC@1={m['ACC@1']:.3f}  F1@1={m['F1@1']:.3f}  ACC@{top_k}={acc3_str}")
            
            print("High-confidence predictions (first 5 val images):")
            for img_obj in hi_preds[:5]:
                print(f"  {img_obj['file']}:")
                for t, pairs in img_obj["pred"].items():
                    txt = ", ".join(f"{lbl}:{pr:.2f}" for lbl, pr in pairs[:5])
                    print(f"    {t}: {txt}")

            # -------- 保存 JSON --------
            json_path = out_dir / f"val_preds_fold{fold}_ep{ep}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(hi_preds, f, ensure_ascii=False, indent=2)


            torch.save({
                'model':model.state_dict(),
                'backbone':args.backbone,
                'num_classes':nc,
                'epoch':ep,'val_loss':val_loss,
                'metrics':metrics}, 
                out_dir/f'epoch{ep}.pt')

# ------------------ main ------------------

def main():
    args=parse_args()
    # if args.mix_prob + args.cutmix_prob > 1.0:
    #     raise ValueError("mix_prob + cutmix_prob must be ≤ 1.0")
    if args.tasks != 'tongue' and (args.mix_prob + args.cutmix_prob > 1.0):
        raise ValueError("mix_prob + cutmix_prob must be ≤ 1.0")
    task = args.tasks
    cv_dir = f'cv_splits/{task}'
    os.makedirs(cv_dir, exist_ok=True)
   
    # base_ds = get_dataset(args)
    if args.tasks == 'tongue':
        base_ds = MultiViewTongueDataset(args.csv, args.img_root, is_train=False, img_size=224, ra_prob=0.0)
    else: 
        base_ds = get_dataset(args)
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.cv > 1:
        # 1) If no splits on disk yet, generate & save them
        if not any(fname.startswith('fold_') for fname in os.listdir(cv_dir)):
            kf = KFold(n_splits=args.cv, shuffle=True, random_state=42)
            for i, (tr_idx, va_idx) in enumerate(kf.split(range(len(base_ds))), 1):
                save_split(cv_dir, i, tr_idx, va_idx, base_ds)

        # 2) Load each fold from its JSON and train
        for fname in sorted(os.listdir(cv_dir)):
            if not fname.startswith('fold_') or not fname.endswith('.json'):
                continue
            fold_id = fname.split('_')[1].split('.')[0]
            split = json.load(open(os.path.join(cv_dir, fname), 'r'))
            tr_idx = split['train_idx']
            va_idx = split['val_idx']
            train_fold(tr_idx, va_idx, fold_id, args)
    else:
        tr,va = train_test_split(
             range(len(base_ds)),
             test_size=0.2,
             random_state=42,
             shuffle=True)
        # if you want to save the hold‑out split as “fold_0.json”:
        save_split(cv_dir, 0, tr, va, base_ds)
        train_fold(tr, va, '0', args)

if __name__=='__main__':
    main()
