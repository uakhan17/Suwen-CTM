#!/usr/bin/env bash
set -euo pipefail          # stop on first error, undefined var, or pipe failure

export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/miniforge3}/lib:${LD_LIBRARY_PATH:-}"
python train_tongue_base2.py \
  --csv  data/tongue/dataset_tongue.csv \
  --img_root data/tongue/images \
  --tasks tongue \
  --backbone convnext_tiny \
  --batch 64 \
  --epochs 50 \
  --cv 5 \
  --lr 5e-4 \
  --cutmix_prob 0 \
  --gradnorm \
  --clip 0.5 \
  --device cuda:1 \
  --out_dir ckp/tongue/convnext_noaug_base0727_bsloss \
  --mix_prob 0 \
  --beta 0