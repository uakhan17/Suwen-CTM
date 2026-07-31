#!/usr/bin/env bash
set -euo pipefail          # stop on first error, undefined var, or pipe failure

export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/miniforge3}/lib:${LD_LIBRARY_PATH:-}"
python train_tongue_base.py \
  --csv  data/tongue/dataset_tongue.csv \
  --img_root data/tongue/images \
  --tasks tongue \
  --backbone convnext_tiny \
  --batch 64 \
  --epochs 50 \
  --lr 5e-4 \
  --cutmix_prob 0 \
  --gradnorm \
  --clip 0.5 \
  --device cuda:1 \
  --out_dir ckp/tongue/convnext_tongue_fulldata \
  --mix_prob 0 \
  --beta 0 \
  --full
  # --cv 5 \

# helper to avoid duplication
# run () {
#   local tag=$1 ra=$2 mix=$3 cut=$4 beta=$5
#   echo "===== $tag ====="
#   python train.py \
#     --csv data/tongue/dataset_tongue.csv \
#     --img_root data/tongue/images \
#     --tasks tongue \
#     --backbone convnext_tiny \
#     --batch 64 \
#     --epochs 100 \
#     --lr 5e-4 \
#     --cv 5 \
#     --gradnorm \
#     --clip 0.5 \
#     --device cuda:0 \
#     --out_dir "ckp/tongue/${tag}" \
#     --ra_prob "$ra" \
#     --mix_prob "$mix" \
#     --cutmix_prob "$cut" \
#     --beta "$beta"
# }

# run "convnext_noaug" 0   0    0    0
# run "convnext_rand"  0.5 0    0    0
# run "convnext_mixup" 0   0.5  0    0.4
# run "convnext_cutmix" 0  0    0.5  0.4
# run "convnext_allaug" 0.33 0.33 0.34 0.4   # sum ≤ 1

# for backbone in resnet50 densenet121 efficientnet_b0; do
#   echo
#   echo "===== Backbone: $backbone  (ra=50%, mixup vs cutmix always) ====="
#   python train.py \
#     --csv data/tongue/dataset_tongue.csv \
#     --img_root data/tongue/images \
#     --tasks tongue \
#     --backbone $backbone \
#     --batch 64 \
#     --epochs 100 \
#     --lr 1e-5 \
#     --cv 5 \
#     --gradnorm \
#     --clip 0.5 \
#     --device cuda:1 \
#     --out_dir ckp/tongue/${backbone}_allaug \
#     --ra_prob 0.5 \
#     --mix_prob 0.5 \
#     --cutmix_prob 0.5 \
#     --beta 0.4
# done

# echo "Done."




# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# python train.py \
# --csv data/tongue/dataset_tongue.csv \
# --img_root data/tongue/images \
# --tasks tongue \
# --backbone resnet50 \
# --batch 64 \
# --epochs 100 \
# --lr 1e-5 \
# --cv 5 \
# --gradnorm \
# --clip 0.5 \
# --device cuda:0 \
# --ra_prob 0.5 \
# --mix_prob 0 \
# --cutmix_prob 0 \
# --out_dir ckp/tongue/resnet50

# python train.py \
# --csv data/tongue/dataset_tongue.csv \
# --img_root data/tongue/images \
# --tasks tongue \
# --backbone densenet121 \
# --batch 64 \
# --epochs 100 \
# --lr 1e-5 \
# --cv 5 \
# --gradnorm \
# --clip 0.5 \
# --device cuda:0 \
# --ra_prob 0.5 \
# --mix_prob 0 \
# --cutmix_prob 0 \
# --out_dir ckp/tongue/densenet121

# python train.py \
# --csv data/tongue/dataset_tongue.csv \
# --img_root data/tongue/images \
# --tasks tongue \
# --backbone efficientnet_b0 \
# --batch 64 \
# --epochs 100 \
# --lr 1e-5 \
# --cv 5 \
# --gradnorm \
# --clip 0.5 \
# --device cuda:0 \
# --ra_prob 0.5 \
# --mix_prob 0 \
# --cutmix_prob 0 \
# --out_dir ckp/tongue/efficientnet_b0

# # for arch in resnet50 densenet121 efficientnet_b0; do
# #     python train.py \
# #       --csv data/tongue/dataset_tongue.csv \
# #       --img_root data/tongue/images \
# #       --tasks tongue \
# #       --backbone "$arch" \
# #       --batch 64 \
# #       --epochs 100 \
# #       --lr 1e-5 \
# #       --cv 5 \
# #       --gradnorm \
# #       --clip 0.5 \
# #       --device cuda:0 \
# #       --out_dir "ckp/tongue/$arch"
# # done
#!/usr/bin/env bash
# set -e  # exit on first error

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# # … your randaug / baseline blocks above …

# # 1) MixUp only
# echo "==============================="
# echo "STARTING MIXUP EXPERIMENTS"
# echo "==============================="
# for backbone in resnet50 densenet121 efficientnet_b0; do
#   echo "  → mixup on $backbone"
#   python train.py \
#     --csv data/tongue/dataset_tongue.csv \
#     --img_root data/tongue/images \
#     --tasks tongue \
#     --backbone $backbone \
#     --batch 64 \
#     --epochs 100 \
#     --lr 1e-5 \
#     --cv 5 \
#     --gradnorm \
#     --clip 0.5 \
#     --device cuda:0 \
#     --out_dir ckp/tongue/${backbone}_mixup \
#     --ra_prob 0 \
#     --mix_prob 0.5 \
#     --beta 0.4 \
#     --cutmix_prob 0
# done

# # 2) CutMix only
# echo "==============================="
# echo "STARTING CUTMIX EXPERIMENTS"
# echo "==============================="
# for backbone in resnet50 densenet121 efficientnet_b0; do
#   echo "  → cutmix on $backbone"
#   python train.py \
#     --csv data/tongue/dataset_tongue.csv \
#     --img_root data/tongue/images \
#     --tasks tongue \
#     --backbone $backbone \
#     --batch 64 \
#     --epochs 100 \
#     --lr 1e-5 \
#     --cv 5 \
#     --gradnorm \
#     --clip 0.5 \
#     --device cuda:0 \
#     --out_dir ckp/tongue/${backbone}_cutmix \
#     --ra_prob 0 \
#     --mix_prob 0 \
#     --beta 0.4 \
#     --cutmix_prob 0.5
# done

# echo "ALL MIXUP & CUTMIX JOBS FINISHED!"

# set -e  # exit on first error

