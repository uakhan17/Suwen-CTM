# 🌿 Suwen-CTM: Intelligent Traditional Chinese Medicine Diagnostic & Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Suwen-CTM** is a end-to-end, production-deployed AI system designed for automated **Chinese Traditional Medicine (CTM)** diagnosis and prescription recommendation. By combining multitask vision architectures (ConvNeXt v2), custom computer-vision quality audit gates, and hybrid NLP retrieval engines (BM25 + BGE Embeddings), Suwen-CTM translates visual face and tongue features and summary of chief complaint into actionable, diagnostic insights.

---

## ⚡ Key Engineering Challenges & Solutions

Developing AI for CTM domain application presents unique real-world technical hurdles:

> ### 🎯 Challenge 1: Limited Training Data Samples
> * **The Problem:** Clinical CTM datasets with expert annotations are inherently scarce and expensive to acquire.
> * **Our Solution:** Implemented **ConvNeXt v2** backbone transfer learning paired with an aggressive data augmentation triad (**RandAugment + MixUp + CutMix**) and **5-Fold Cross-Validation with Exponential Moving Average (EMA)** smoothing to prevent overfitting while maximizing representation feature retention.

> ### 📊 Challenge 2: 118 Fine-Grained & Highly Imbalanced Diagnostic Labels
> * **The Problem:** The taxonomy contains **118 fine-grained labels** across 19 sub-tasks (e.g., specific tongue coat textures, facial luster, lip color variations), leading to a severe long-tail class distribution.
> * **The Clinical Upside:** Unlike generic vision labels, these 118 attributes are rigorously defined, mutually consistent, and directly map to CTM diagnostic logic. Uncertainty-aware **Gumbel-Max exploration sampling** is deployed during inference to stabilize decision boundaries for rare tail classes.

> ### 🛡️ Challenge 3: In-the-Wild Real-World Photo Noise
> * **The Problem:** Real-world patient photos submitted via mobile devices suffer from poor lighting, improper mouth opening, off-center framing, and background clutter.
> * **Our Solution:** Engineered custom, two-tier **Visual Sanity Check Gates** (`audit_face` & `audit_tongue`). Powered by SCRFD landmark detection, MediaPipe FaceMesh, CIELAB color clustering, and adaptive auto-rotation, these gatekeepers automatically detect and reject invalid inputs before running GPU-intensive inference.

---

## 🏗️ Repository Architecture

This repository is structured as a full-stack AI ecosystem. Click into any subfolder to view its dedicated documentation:

```text
Suwen-CTM/
├── Data Preprocessing/          # EDA scripts, label distribution charts, imbalance mitigation
├── Sanity Check Models/               # Pre-inference quality audit gates (audit_face & audit_tongue)
├── Face & Tongue Diagnosis Models/            # Multi-task ConvNeXt v2 training, 5-Fold CV, weights export
├── Prescription Recommendation System/         # Hybrid BM25 + BGE retrieval, safety guard, Gradio demo
└── Cloud Deployment/            # FastAPI router, Docker containerization, cloud deployment configs
```

---

## 🔄 End-to-End System Pipeline

```text
                     ┌───────────────────────────────────────────┐
                     │    Patient Input (Face / Tongue Photo)    │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │ 1. Visual Sanity Check Gates              │
                     │    - Face: Pose, Yaw/Roll, Center         │
                     │    - Tongue: Mouth Gap (MAR), Color Blob  │
                     └─────────────┬───────────────────┬─────────┘
                                   │                   │
                          [FAILED] │                   │ [PASSED]
                                   ▼                   ▼
           ┌──────────────────────────────┐   ┌──────────────────────────────┐
            ❌ Early Rejection Warning         2. Multitask Neural Network  
                                                   
           └──────────────────────────────┘   └──────────────┬───────────────┘
                                                             │ 19 Diagnostic Labels
                                                             | +
                                                             | Summary of Chief Complaint
                                                             ▼
           ┌──────────────────────────────┐   ┌──────────────────────────────┐
           │ 4. Interactive Gradio Demo   │   │ 3. Hybrid Prescription Engine
           │    & Production FastAPI      │◀──  - BM25 + BGE Dense Retrieval
           │    (/predict Endpoint)       │   │ - 18 Fan / 19 Wei Safety Guard
           └──────────────────────────────┘   └──────────────────────────────┘
```

---

## 🧩 Core Sub-Modules Overview

| Module Directory | Key Technologies | Description |
| :--- | :--- | :--- |
| [**`Data_Preprocessing/`**](./Data_Preprocessing/) | Pandas, Matplotlib, Seaborn | Analyzes 19 target feature distribution bar charts and handles dataset cleaning. |
| [**`Sanity_Checks/`**](./Sanity_Checks/) | OpenCV, SCRFD, MediaPipe, CIELAB | Pre-inference guardrails that filter out invalid photos, extreme roll angles, or closed mouths. |
| [**`Vision_Diagnosis/`**](./Vision_Diagnosis/) | PyTorch, ConvNeXt v2, timm | Multi-task vision models trained with RandAugment, CutMix, MixUp, and 5-Fold Cross-Validation. |
| [**`Prescription_System/`**](./Prescription_System/) | Rank-BM25, BGE-Small, Gradio | Hybrid CTM prescription retrieval engine with automated **Eighteen Incompatibilities (十八反)** safety interception. |
| [**`Cloud_Deployment/`**](./Cloud_Deployment/) | FastAPI, Uvicorn, Docker, ONNX | Containerized deployment backend with Gumbel-Max sampling and healthcheck monitoring. |

---
