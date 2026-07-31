# 🛡️ Visual Sanity Check Models (Quality Audit Gates)

To prevent garbage-in-garbage-out failure modes in downstream multi-task diagnosis, this directory implements pre-inference **quality gating engines** for face (`audit_face.py`) and tongue (`audit_tongue.py`) images. 

These lightweight sanity checks ensure incoming patient photos meet strict CTM (Traditional Chinese Medicine) diagnostic standards before incurring GPU inference overhead.

---

## 📐 1. Face Quality Gate Logic (`audit_face.py`)

The face audit engine evaluates image framing, pose angles, facial symmetry, and subject prominence.

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                    Input Patient Image                      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Multi-Strategy SCRFD Detection                           │
  │    (Multi-scale, Border Padding, CLAHE/Gamma Contrast)      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                    [Failed]     ▼     [Passed]
          ┌──────────────────────┴──────────────────────┐
          │ MediaPipe FaceDetection Fallback (CPU)      │
          └──────────────────────┬──────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. Primary Face Selection & Multi-Face Rejection             │
  │    Score = Conf × (0.6 + 0.4×(1 - CenterNorm)) × √(Area)   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. Geometric & Pose Audits                                  │
  │    - Area Bounds: [AREA_MIN, AREA_MAX]                      │
  │    - Yaw Asymmetry: Eye-to-Nose ratio <= MAX_YAW_ASYM        │
  │    - Roll Angle: Eye-level tilt <= MAX_ROLL_DEG (20°)       │
  │    - Plausibility: Eye-nose-mouth vertical ratio & spacing  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
    [FAIL] ──> Reject with Diagnostic Warning Message
    [PASS] ──> Accept Payload for Multi-Task Inference
```

### Key Audit Components

* **Multi-Strategy Detection (`detect_face_robust`):** Sweeps images through multiple resolutions (`640x640`, `960x960`, `320x320`), downscale ratios, border padding, CLAHE contrast enhancement, and Gamma transformations to recover valid faces under extreme lighting or distance.
* **MediaPipe Fallback:** If SCRFD ONNX fails, a lightweight MediaPipe Face Detection fallback executes to recover close-up features while maintaining false-positive rejection on background artifacts.
* **Prominence-Based Primary Selection (`select_primary`):** Calculates a weighted score using confidence, bounding box area, and distance from image center. If secondary faces with significant area ($\ge 0.01$) and confidence ($\ge 0.50$) are detected, the system issues a multi-face rejection warning.
* **Pose & Symmetry Validation:**
  * **Yaw Asymmetry:** Measures left/right eye-to-nose distance ratios ($|d_r - d_l| / \max(d_r + d_l)$) to reject extreme side-profiles.
  * **Roll Angle:** Evaluates eye-line inclination, triggering automatic rotation auto-fix or rejection if tilt exceeds $20^\circ$.
  * **Plauisibility Checks:** Enforces proportional distance ratios between eyes, nose tip, and mouth corners.

---

## 👅 2. Tongue Quality Gate Logic (`audit_tongue.py`)

Tongue inspection requires distinct quality criteria: the mouth must be sufficiently open, the tongue body clearly protruded, and diagnostic surface characteristics (coat color, fissures) unobscured.

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                    Input Tongue Image                       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Stage 1: MediaPipe FaceMesh Outer/Inner Lip Landmarks        │
  │  - Calculate MAR (Mouth Aspect Ratio = Inner Gap / Width)   │
  │  - Extract ROI around lip region (padded)                   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                   [Success]     ▼     [No FaceMesh]
          ┌──────────────────────┴──────────────────────┐
          │ Stage 2: SCRFD Lower-Face Lower ROI Crop     │
          └──────────────────────┬──────────────────────┘
                                 │
                   [Success]     ▼     [No SCRFD]
          ┌──────────────────────┴──────────────────────┐
          │ Stage 3: Central Reddish/Coated Blob Filter │
          └──────────────────────┬──────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Color & Morphological Verification (CIELAB Space)           │
  │  - Multi-Color Masking: Pink/Red, White/Gray, Purple, Yellow│
  │  - Protrusion Metric: Distance past lip-corner baseline     │
  │  - Contour Solidity: Convexity check (Solidity >= 0.58)     │
  └─────────────────────────────────────────────────────────────┘
```

### Key Audit Components

* **Multi-Stage Tiered Cascade:**
  1. **Primary (`MediaPipe FaceMesh`):** Extracts precise 3D facial landmarks for outer/inner lips. Calculates the **Mouth Aspect Ratio (MAR)**:
     $$\text{MAR} = \frac{\|\text{Lower Inner Lip} - \text{Upper Inner Lip}\|}{\|\text{Right Corner} - \text{Left Corner}\|}$$
  2. **Secondary (`SCRFD Lower-Face`):** If full face mesh fails, crops the lower $55\%$ region of an SCRFD-detected face bounding box.
  3. **Fallback (`Central Blob Search`):** For extreme close-ups showing only the oral cavity, isolates large, central, high-solidity color blobs.
* **CTM Multi-Cluster CIELAB Masking:** Identifies distinct tongue diagnostic color features:
  * **Classic Red/Pink:** Standard tongue tissue ($a^* \in [135, 185]$, $b^* \in [120, 180]$).
  * **White/Gray Coat:** Thick coating ($L^* > 155$, restricted $a^*, b^*$).
  * **Purple/Stasis:** Microcirculation stasis ($a^* \in [140, 190]$, lower $b^* \in [90, 120]$).
  * **Yellow/Damp-Heat:** Damp-heat coating ($a^* \in [110, 140]$, elevated $b^* \in [145, 195]$).
* **Protrusion Depth & Contour Solidity:** Calculates the normalized depth of the tongue mask relative to the lip-corner baseline ($\text{ProtrudeNorm} \ge 0.16$). Filters thin lip lines using contour-to-convex-hull area ratios ($\text{Solidity} \ge 0.58$).

---

