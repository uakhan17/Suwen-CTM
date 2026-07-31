FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    TQDM_DISABLE=1

# Set the DEBIAN_FRONTEND to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps:
# - curl for healthcheck
# - libgl/libglib for OpenCV (headless)
# - build-essential (g++) to build insightface's face3d cython extension if a wheel is not available
# - cmake, git are harmless and sometimes required by pip builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 \
    build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

# Keep pip toolchain recent to avoid wheel build quirks
RUN pip install --no-cache-dir -U pip setuptools wheel

# Python deps
# - torchvision pinned to match torch 2.1.0 (CUDA 11.8)
# - opencv-python-headless (no GUI libs)
# - insightface (provides SCRFD wrapper) — needs g++
# - onnxruntime-gpu pinned to 1.16.3 which matches CUDA 11.8
# - mediapipe for fallback logic in gates
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    fastapi uvicorn[standard] pydantic pillow \
    torchvision==0.16.0 timm python-multipart \
    opencv-python-headless insightface==0.7.3 "onnxruntime-gpu==1.16.3" mediapipe absl-py

# Code & weights baked into the image
COPY app_router.py model.py audit_face.py audit_tongue.py /app/
COPY face_ema.pt tongue_ema.pt scrfd_2.5g_bnkps.onnx /app/

# Default ENV for weights baked into image
ENV FACE_WEIGHTS=/app/face_ema.pt \
    TONGUE_WEIGHTS=/app/tongue_ema.pt \
    SCRFD_ONNX=/app/scrfd_2.5g_bnkps.onnx

# ---------- Aggressive safeguard defaults (override with -e) ----------
ENV EXPLORE_ON=1 \
    EXPLORE_MODE=full \
    TOPN_POOL=7 \
    SAMPLE_T=1.5 \
    UNCERTAINTY_METRIC=margin \
    MARGIN_TH=1.0 \
    ENTROPY_TH=1.2 \
    MIN_PROB=0.10 \
    PROB_CAP=0.85 \
    INFER_TTA=0 \
    SEED_SALT=ctm_salt_v1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# Use one worker to keep GPU memory simple/predictable
CMD ["uvicorn", "app_router:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]