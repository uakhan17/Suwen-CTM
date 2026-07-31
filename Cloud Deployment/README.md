# ☁️ Cloud Deployment & Service Architecture

This directory handles the containerized deployment of the **Suwen-CTM Inference Service** on Alibaba Cloud, powered by FastAPI, PyTorch, and ONNX Runtime.

---

## 1. Service Logic & Data Flow

The core entry point is `app_router.py`, a high-performance FastAPI application that receives visual data (face or tongue images), performs pre-inference quality gating, routes images through task-specific multi-head neural networks, and handles deterministic prediction sampling.

```text
                  ┌──────────────────────────────────────────────┐
                  │          HTTP POST Request (/predict)        │
                  │   (Form-Data Upload OR Base64 JSON Batch)    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    1. Image Decoding & Tensor Conversion     │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     2. Quality Audit Gate (SCRFD ONNX)       │
                  │       - Detect Face/Tongue Landmarks         │
                  │       - Auto-Rotate Fix (if small roll angle)│
                  └──────────────┬────────────────┬──────────────┘
                                 │                │
                        [FAILED] │                │ [PASSED]
                                 ▼                ▼
         ┌──────────────────────────────┐   ┌──────────────────────────────┐
         │ ❌ Hard Rejection Output     │   │ 3. Multitask Neural Forward  │
         │ "采集信息有误，请重拍照片"    │   │   - ImageNet Normalization   │
         └──────────────────────────────┘   │   - Optional TTA (Flip)      │
                                            │   - Multi-Head Logit Output  │
                                            └──────────────┬───────────────┘
                                                           │
                                                           ▼
                                            ┌──────────────────────────────┐
                                            │ 4. Uncertainty & Sampling    │
                                            │   - Margin / Entropy Threshold│
                                            │   - Gumbel Top-K Exploration │
                                            └──────────────┬───────────────┘
                                                           │
                                                           ▼
                                            ┌──────────────────────────────┐
                                            │  ✅ JSON Prediction Response │
                                            └──────────────┬───────────────┘
```

### Detailed Pipeline Breakdown

#### 1. Lifecycle Initialization (`lifespan`)
When the FastAPI application initializes on startup:
* **Model Pack Loading:** Loads weights into CPU/GPU memory for both the `face` model (13 classification task heads) and the `tongue` model (6 classification task heads) using `MultiTaskNet`.
* **Shared Gate Model (SCRFD):** Loads the `scrfd_2.5g_bnkps.onnx` face detection model via ONNX Runtime to perform landmark verification and quality checks.

#### 2. Request Handling & Ingestion
The `/predict` endpoint accepts two types of payloads:
* **Form-Data:** Single image file upload paired with target parameters (`model`: `face` or `tongue`, `topk`, `request_id`).
* **JSON Base64:** Single or multi-image batch payloads containing base64-encoded strings with dedicated `request_id` tracking.

#### 3. Pre-Inference Quality Gate (`audit_face` & `audit_tongue`)
Before passing an image to the deep learning model:
* **Landmark Gating:** The image is checked against SCRFD landmarker constraints to ensure valid face/tongue positioning.
* **Auto-Fix Mechanism:** For face images failing roll-angle tolerances (up to `AUTO_ROTATE_MAX_DEG`), the service automatically calculates central affine rotation matrices, adjusts the image alignment, and re-audits.
* **Hard Rejection:** If gating fails completely, inference is bypassed immediately, returning a standard rejection response: `{"warning": "采集信息有误，请重拍照片", "results": []}`.

#### 4. Multitask Model Forward Pass & TTA
* **Preprocessing:** Valid images are converted, resized to model dimension (e.g., 224×224), and normalized using standard ImageNet mean and standard deviation.
* **Test-Time Augmentation (TTA):** If `INFER_TTA` is enabled, the model evaluates both the original and horizontally flipped tensors, averaging prediction logits across task heads.

#### 5. Uncertainty Metric & Gumbel Exploration
To improve diagnosis recommendations in ambiguous edge cases:
* **Uncertainty Check:** Logits are evaluated using Margin (`_logit_margin`) or Entropy (`_entropy_from_logits`) against user-configured thresholds (`MARGIN_TH`, `ENTROPY_TH`).
* **Exploration Sampling:** When uncertainty is high and `EXPLORE_ON` is active, a deterministic pseudo-random Gumbel-Max trick (salted with `SEED_SALT` + `request_id`) samples top-K candidate labels from the top pool, preventing static lock-in on uncertain boundaries while maintaining idempotency per request ID.

---

## 2. Dockerfile Explained

The `Dockerfile` provides a fully self-contained, GPU-accelerated container image tailored for high-throughput PyTorch and ONNX Runtime execution on Alibaba Cloud container instances (e.g., ACK or Serverless Container Instance).

---

### Layer-by-Layer Architectural Breakdown

#### 1. Base Image & System Environment
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    TQDM_DISABLE=1 \
    DEBIAN_FRONTEND=noninteractive
```
* **Base Image:** Built on official PyTorch 2.1.0 with CUDA 11.8 and cuDNN 8 runtime support for GPU hardware acceleration.
* **Python Runtime Flags:** Disables `.pyc` bytecode creation and unbuffers `stdout`/`stderr` streams so log output appears instantly in container log aggregators.
* **System Settings:** Configures timezone to `Asia/Shanghai` and silences interactive prompts during system package installations.

---

#### 2. System Dependencies
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 \
    build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*
```
* **Headless Graphics Libraries (`libgl1`, `libglib2.0-0`, etc.):** Required for OpenCV (`opencv-python-headless`) frame manipulation without needing GUI desktop components.
* **Compilation Toolchain (`build-essential`, `cmake`):** Provides C/C++ compilers (`g++`) required to compile `insightface` Cython extensions (like `face3d`).
* **Utility Libraries (`curl`):** Used by the Docker engine to perform internal healthchecks.

---

#### 3. Python Package Stack
```dockerfile
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    fastapi uvicorn[standard] pydantic pillow \
    torchvision==0.16.0 timm python-multipart \
    opencv-python-headless insightface==0.7.3 "onnxruntime-gpu==1.16.3" mediapipe absl-py
```
* **Strict Version Pinning:** Matches `torchvision==0.16.0` with PyTorch 2.1.0 and pins `onnxruntime-gpu==1.16.3` for CUDA 11.8 compatibility.
* **Numpy Guard (`numpy<2.0`):** Prevents ABI breakage from NumPy 2.0 release across C-extensions (like OpenCV and InsightFace).
* **Web & ML Tools:** Installs FastAPI/Uvicorn ASGI web framework and MediaPipe for fallback landmark detection logic.

---

#### 4. Baked Code & Model Weights
```dockerfile
COPY app_router.py model.py audit_face.py audit_tongue.py /app/
COPY face_ema.pt tongue_ema.pt scrfd_2.5g_bnkps.onnx /app/

ENV FACE_WEIGHTS=/app/face_ema.pt \
    TONGUE_WEIGHTS=/app/tongue_ema.pt \
    SCRFD_ONNX=/app/scrfd_2.5g_bnkps.onnx
```
* **Air-Gapped Container Image:** All application code files and trained model weight checkpoints (`face_ema.pt`, `tongue_ema.pt`, `scrfd_2.5g_bnkps.onnx`) are copied directly into the image layer at build time.
* **Zero External Fetching:** Eliminates runtime network downloading of weights, guaranteeing zero startup dependency delays.

---

#### 5. Default Runtime Safeguard Variables
```dockerfile
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
```
* Sets production safety defaults for logit capping, uncertainty margin triggers, and Gumbel-Max exploration sampling. 
* *Note: All environment variables can be overridden dynamically at container launch via `docker run -e` or Kubernetes deployment configs.*

---

#### 6. Healthcheck & Process Execution
```dockerfile
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS [http://127.0.0.1:8000/healthz](http://127.0.0.1:8000/healthz) || exit 1

CMD ["uvicorn", "app_router:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```
* **Port Mapping:** Exposes port `8000` for FastAPI traffic.
* **Automated Healthcheck:** Periodically queries `/healthz` every 30 seconds to ensure the model service and GPU runtime context are healthy.
* **Single Worker Process Execution (`--workers 1`):** Launches Uvicorn with a single worker process. This is critical for PyTorch GPU deployments to prevent multiple worker processes from duplicating VRAM allocations on the same GPU.