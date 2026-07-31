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
from datasets import TongueDataset, FaceDataset, collate_multitask, ALL_LABELS
from model import MultiTaskNet
from tqdm import tqdm
from pathlib import Path
import numpy as np
# --------------------------- CLI ---------------------------

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
    p.add_argument('--full', action='store_true', help='Train on 100% of data (no validation split).')
    return p.parse_args()

def get_dataset(args):
    ds_kwargs = dict(ra_prob=args.ra_prob)  # Dataset handles RandAug
    if args.tasks == 'tongue':
        return TongueDataset(args.csv, args.img_root, **ds_kwargs)
    return FaceDataset(args.csv, args.img_root, **ds_kwargs)


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
    base_ds = get_dataset(args)
    tr_ds, va_ds = Subset(base_ds, tr_idx), Subset(base_ds, va_idx)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate_multitask)
    if len(va_idx) == 0:
        va_loader = None
    else:
        va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_multitask)
    # va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_multitask)

    model = MultiTaskNet(base_ds.nc, backbone=args.backbone).to(args.device)
    # params_backbone = list(model.backbone.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # GradNorm init
    task_names = list(base_ds.nc)
    W = torch.ones(len(task_names), device=args.device, requires_grad=False)
    alpha, L0 = 1.5, None

    out_dir = pathlib.Path(args.out_dir)/f'fold{fold}'; out_dir.mkdir(parents=True, exist_ok=True)

    for ep in tqdm(range(1, args.epochs+1),colour='green'):
        model.train() 
        ep_loss=0 
        t0=time.time()
        iter_losses = []                                   # ← 新建列表
        pbar = tqdm(tr_loader, desc=f"Fold {fold} Ep {ep}", ncols=100)
        
        for step, (imgs, gts, fnames) in enumerate(pbar):
            imgs = imgs.to(args.device)
            gts  = {k:v.to(args.device) for k,v in gts.items()}

            # ---------------- batch‑level MixUp / CutMix ----------------
            r = random.random()
            if r < args.mix_prob:
                lam = torch.distributions.Beta(args.beta, args.beta).sample().item()
                idx = torch.randperm(imgs.size(0)).to(args.device)
                imgs = lam*imgs + (1-lam)*imgs[idx]
                tgt_b = {k:v[idx] for k,v in gts.items()}
            elif r < args.mix_prob + args.cutmix_prob:
                lam = torch.distributions.Beta(args.beta, args.beta).sample().item()
                idx = torch.randperm(imgs.size(0)).to(args.device)
                tgt_b = {k:v[idx] for k,v in gts.items()}
                x1,y1,x2,y2 = rand_bbox(imgs.size(3), imgs.size(2), lam)
                imgs[:,:,y1:y2,x1:x2] = imgs[idx,:,y1:y2,x1:x2]
                lam = 1 - ((x2-x1)*(y2-y1)/(imgs.size(-1)*imgs.size(-2)))
            else:
                lam, tgt_b = 1.0, None

            preds = model(imgs)

            task_losses = []
            task_losses_tmp = []
            for t in task_names:
                n_cls = base_ds.nc[t]                           # 类别数

                # ── 把整数标签变成 one-hot ─────────────────────────
                y_a = F.one_hot(gts[t], n_cls).float()          # [B, C]
                if tgt_b is None:                               # 无 MixUp/CutMix
                    target = y_a                                # 软标签 (仍是 one-hot)
                else:                                           # MixUp 或 CutMix
                    y_b = F.one_hot(tgt_b[t], n_cls).float()
                    target = lam * y_a + (1 - lam) * y_b        # [B, C] soft-label

                # ── soft-label Cross-Entropy (PyTorch ≥1.10 支持) ─
                loss = F.cross_entropy(preds[t], target, reduction="mean")
                task_losses.append(loss)
                task_losses_tmp.append((t, loss))

            task_losses = torch.stack(task_losses)
            task_losses_dict = {name: loss for name, loss in task_losses_tmp}
            # -------- GradNorm --------
            # if args.gradnorm:
            #     if L0 is None:
            #         L0 = task_losses.detach()
            #     weighted = (W*task_losses).sum()
            #     opt.zero_grad(); weighted.backward(retain_graph=True)
            #     # gradient norms via head weight
            #     grad_norm = torch.stack([model.heads[t].weight.grad.norm() for t in task_names])
            #     G_avg = grad_norm.mean()
            #     inv_rate = (task_losses.detach()/L0)
            #     target = (inv_rate/inv_rate.mean())**alpha * G_avg
            #     with torch.no_grad():
            #         W *= (grad_norm/target)
            #         W = (W/W.mean()).detach()
            # else:
            #     weighted = task_losses.sum(); opt.zero_grad(); weighted.backward()
            eps = 1e-8                         # <‑‑ single global constant is enough

            if args.gradnorm:
                if L0 is None:
                    # store baseline magnitudes once; add eps so zero‑loss heads are safe
                    L0 = task_losses.detach() + eps

                # ------------------------------------------------------------------
                weighted = (W * task_losses).sum()
                opt.zero_grad()
                weighted.backward(retain_graph=True)

                # 1) gradient norms for each head
                grad_norm = torch.stack([
                    model.heads[t].weight.grad.norm() + eps   # +eps so never zero
                    for t in task_names
                ])
                G_avg = grad_norm.mean()

                # 2) inverse training rate
                inv_rate = task_losses.detach() / L0          # denominator ≥ eps
                inv_rate = inv_rate / (inv_rate.mean() + eps) # normalise, eps‑guard
                target   = (inv_rate ** alpha) * G_avg        # eq. (4) in GradNorm

                # 3) update weights
                with torch.no_grad():
                    W *= grad_norm / (target + eps)           # eq. (6) with guard
                    # replace NaN / Inf that might still slip through
                    W = torch.nan_to_num(W, nan=1.0, posinf=1.0, neginf=1.0)
                    W /= W.mean() + eps                       # re‑normalise
            else:
                weighted = task_losses.sum()
                opt.zero_grad()
                weighted.backward()

            # if args.gradnorm:
            #     weighted = safe_gradnorm(task_losses_dict, params_backbone)
            # else:
            #     weighted = torch.stack(list(task_losses.values())).sum()
            # opt.zero_grad(); weighted.backward()
            
            if args.clip>0: 
                clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            iter_loss = weighted.item()
            ep_loss  += iter_loss * imgs.size(0)
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
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'face_Ep{ep}_full_data.pt'))

        if ep%10 == 0 and (va_loader is not None):
            model.eval()
            val_loss = 0.0

            logit_task = {t: [] for t in task_names}   # 收集 raw logits
            truth_task = {t: [] for t in task_names}

            hi_preds: list[dict] = []                 # 每张图像的高置信结果

            with torch.no_grad():
                for imgs, gts, fnames in va_loader:   # collate 返回 (imgs, labels, filenames)
                    imgs = imgs.to(args.device)
                    gts  = {k: v.to(args.device) for k, v in gts.items()}

                    outs = model(imgs)                # dict{t: logits [B, C]}

                    # -------- 累计 val_loss 与指标 --------
                    for t in task_names:
                        val_loss += F.cross_entropy(outs[t], gts[t],reduction="sum"
                                                    ).item()              # 累加总和
                        logit_task[t].append(outs[t].cpu())
                        truth_task[t].append(gts[t].cpu())

                    # -------- 逐图片高置信 --------
                    B = imgs.size(0)
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
                f1_top1  = f1_score(y_true.numpy(),
                                    y_pred.numpy(),
                                    average='macro',
                                    zero_division=0)

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
            # for t, m in metrics.items():
            #     print(f"  {t}: ACC={m['ACC']:.3f} F1={m['F1']:.3f} ACC@3={m['ACC@3']:.3f}")
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
                'num_classes':base_ds.nc,
                'epoch':ep,'val_loss':val_loss,
                'metrics':metrics}, 
                out_dir/f'epoch{ep}.pt')

# ------------------ main ------------------

def main():
    args=parse_args()
    # sanity prob sum
    # total_prob=args.ra_prob+args.mix_prob+args.cutmix_prob
    # if total_prob>1.0:
    #     raise ValueError('ra_prob+mix_prob+cutmix_prob must <=1')
    if args.mix_prob + args.cutmix_prob > 1.0:
        raise ValueError("mix_prob + cutmix_prob must be ≤ 1.0")

    task = args.tasks
    cv_dir = f'cv_splits/{task}'
    os.makedirs(cv_dir, exist_ok=True)
    base_ds = get_dataset(args)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.full:
        tr_idx = list(range(len(base_ds)))
        va_idx = []                      # empty → no validation
        train_fold(tr_idx, va_idx, 'all', args)
        return
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
