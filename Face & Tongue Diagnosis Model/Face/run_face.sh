#!/usr/bin/env bash
set -euo pipefail          # stop on first error, undefined var, or pipe failure

export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/miniforge3}/lib:${LD_LIBRARY_PATH:-}"
python train_face.py \
  --csv  data/face/dataset_face.csv \
  --img_root data/face/images \
  --tasks face \
  --backbone convnext_tiny \
  --batch 64 \
  --epochs 50 \
  --lr 5e-4 \
  --cutmix_prob 0 \
  --clip 0.5 \
  --device cuda:0 \
  --out_dir ckp/face/convnext_face_fulldata \
  --mix_prob 0 \
  --beta 0 \
  --full
  # --gradnorm \
  # --cv 5 \

# #!/usr/bin/env bash
# set -euo pipefail          # stop on first error, undefined var, or pipe failure

# export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/miniforge3}/lib:${LD_LIBRARY_PATH:-}"

# # helper to avoid duplication
# run () {
#   local tag=$1 ra=$2 mix=$3 cut=$4 beta=$5
#   echo "===== $tag ====="
#   python train_face.py \
#     --csv data/face/dataset_face.csv \
#     --img_root data/face/images \
#     --tasks face \
#     --backbone convnext_tiny \
#     --batch 64 \
#     --epochs 100 \
#     --lr 5e-4 \
#     --cv 5 \
#     --clip 0.5 \
#     --device cuda:1 \
#     --out_dir "ckp/face/${tag}" \
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
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# python train_face.py \
# --csv data/face/dataset_face.csv \
# --img_root data/face/images \
# --tasks face \
# --backbone resnet50 \
# --batch 64 \
# --epochs 100 \
# --lr 1e-5 \
# --cv 5 \
# --gradnorm \
# --clip 0.5 \
# --device cuda:1 \
# --ra_prob 0.5 \
# --mix_prob 0 \
# --cutmix_prob 0 \
# --out_dir ckp/face/resnet50

# python train_face.py \
# --csv data/face/dataset_face.csv \
# --img_root data/face/images \
# --tasks face \
# --backbone densenet121 \
# --batch 64 \
# --epochs 100 \
# --lr 1e-5 \
# --cv 5 \
# --gradnorm \
# --clip 0.5 \
# --device cuda:1 \
# --ra_prob 0.5 \
# --mix_prob 0 \
# --cutmix_prob 0 \
# --out_dir ckp/face/densenet121

# python train_face.py \
# --csv data/face/dataset_face.csv \
# --img_root data/face/images \
# --tasks face \
# --backbone efficientnet_b0 \
# --batch 64 \
# --epochs 100 \
# --lr 1e-5 \
# --cv 5 \
# --gradnorm \
# --clip 0.5 \
# --device cuda:1 \
# --ra_prob 0.5 \
# --mix_prob 0 \
# --cutmix_prob 0 \
# --out_dir ckp/face/efficientnet_b0

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
#   python train_face.py \
#     --csv data/face/dataset_face.csv \
#     --img_root data/face/images \
#     --tasks face \
#     --backbone $backbone \
#     --batch 64 \
#     --epochs 100 \
#     --lr 1e-5 \
#     --cv 5 \
#     --gradnorm \
#     --clip 0.5 \
#     --device cuda:1 \
#     --out_dir ckp/face/${backbone}_mixup \
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
#   python train_face.py \
#     --csv data/face/dataset_face.csv \
#     --img_root data/face/images \
#     --tasks face \
#     --backbone $backbone \
#     --batch 64 \
#     --epochs 100 \
#     --lr 1e-5 \
#     --cv 5 \
#     --gradnorm \
#     --clip 0.5 \
#     --device cuda:1 \
#     --out_dir ckp/face/${backbone}_cutmix \
#     --ra_prob 0 \
#     --mix_prob 0 \
#     --beta 0.4 \
#     --cutmix_prob 0.5
# done

# echo "ALL MIXUP & CUTMIX JOBS FINISHED!"

#!/usr/bin/env bash
# set -euo pipefail

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# for backbone in resnet50 densenet121 efficientnet_b0; do
#   echo
#   echo "===== Backbone: $backbone  (ra=50%, mixup vs cutmix always) ====="
#   python train_face.py \
#     --csv data/face/dataset_face.csv \
#     --img_root data/face/images \
#     --tasks face \
#     --backbone $backbone \
#     --batch 64 \
#     --epochs 100 \
#     --lr 1e-5 \
#     --cv 5 \
#     --gradnorm \
#     --clip 0.5 \
#     --device cuda:1 \
#     --out_dir ckp/face/${backbone}_allaug \
#     --ra_prob 0.5 \
#     --mix_prob 0.5 \
#     --cutmix_prob 0.5 \
#     --beta 0.4
# done

# echo "Done."  
