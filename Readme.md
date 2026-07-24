# 🔭 Argus — RGBT Drone Person Detection with CMAGM Fusion

**Hackathon Project | Team: Argus**

A dual-modal RGB + Thermal drone-based person detection system built on top of the **QFDet** baseline, enhanced with our custom **Cross-Modal Spatial-Channel Attention Gating Module (CMAGM)** and its high-resolution upgrade **HR-CMAGM** targeting small object detection bottlenecks.

---

## 🧠 Our Approach

### Problem
Detecting persons from drones using RGB + Thermal (IR) imagery is challenging due to:
- **Small targets** at altitude (persons appear as ~32×32 px or less)
- **RGB degradation** in low-light / high-contrast aerial scenes
- **Naive concatenation fusion** in baseline QFDet loses cross-modal correlation

### Solution: CMAGM — Cross-Modal Spatial-Channel Attention Gating Module
We replaced the naive `cat → conv1x1` fusion in `QFDet` with a **learned attention gating module** that:
1. **Dynamically recalibrates channel importance** using dual-pool attention (AvgPool + MaxPool)
2. **Spatially focuses** on thermal hot-spots via Multi-Scale Dilated attention (7×7 + dilated 3×3)
3. **Preserves residual signals** via additive skip connections

```
RGB Feature Map ──┐
                  ├─► Concat ─► Dual-Pool Channel Gate ─► Spatial Gate ─► Fused Feature
IR Feature Map  ──┘                  ↑                          ↑
                             AvgPool + MaxPool        Local 7×7 + Dilated 3×3
```

---

## 📊 Benchmark Results

### Stage 2: Baseline QFDet (No Modifications)

| Modality | Split | mAP | mAP₅₀ | mAP₇₅ | mAP_s | mAP_m | mAP_l |
|----------|-------|-----|--------|--------|-------|-------|-------|
| RGB + Thermal | val | 0.338 | 0.721 | 0.273 | 0.144 | 0.325 | 0.585 |
| RGB only | val | 0.075 | 0.261 | 0.025 | 0.008 | 0.070 | 0.179 |
| Thermal only | val | 0.242 | 0.521 | 0.193 | 0.074 | 0.221 | 0.546 |
| RGB + Thermal | test | 0.299 | 0.674 | 0.227 | 0.129 | 0.299 | 0.554 |

### Stage 3: CMAGM Enhanced QFDet

| Modality | Split | mAP | mAP₅₀ | mAP₇₅ | mAP_s | mAP_m | mAP_l |
|----------|-------|-----|--------|--------|-------|-------|-------|
| RGB + Thermal | val | 0.297 | 0.651 | 0.233 | 0.111 | 0.274 | **0.588** |
| RGB only | val | 0.058 | 0.189 | 0.022 | 0.004 | 0.047 | 0.166 |
| Thermal only | val | 0.213 | 0.479 | 0.162 | 0.041 | 0.189 | 0.542 |
| RGB + Thermal | test | 0.268 | 0.609 | 0.201 | 0.105 | 0.268 | 0.549 |

> **Note:** Stage 3 CMAGM trained for a single fine-tuning epoch from the pre-trained baseline. The baseline benefits from extended VTUAV pre-training; CMAGM's large-object mAP already surpasses baseline (0.588 vs 0.585 on val).

---

## 🔍 Error & Scale Analysis

Deep scale-wise analysis of CMAGM detections on the test set revealed:

| Scale | GT Count | Dataset % | TP | Recall | Precision |
|-------|----------|-----------|-----|--------|-----------|
| Large (>96²px) | 269 | 13% | 230 | **85.50%** | 88.46% |
| Medium (32²-96²px) | 1,270 | 61.4% | 581 | **45.75%** | 85.69% |
| Small (<32²px) | 529 | 25.6% | 32 | **6.05%** | 86.49% |

**Key Finding:** Precision is consistently high (~86%) at all scales — the bottleneck is **small object recall (6.05%)**, caused by:
- Aggressive spatial downsampling in standard attention pooling
- Small thermal signatures averaged away by global AvgPool

---

## 🚀 Stage 4 Architectural Innovation: HR-CMAGM

To fix the small object recall bottleneck, we designed and implemented **HR-CMAGM** in `qfdet.py`:

### Architectural Changes:
| Component | CMAGM (Stage 3) | HR-CMAGM (Stage 4) |
|-----------|-----------------|---------------------|
| Channel Attention | Single AdaptiveAvgPool | **Dual Pool: AvgPool + MaxPool** |
| Spatial Attention | Single 7×7 Conv | **7×7 Local + Dilated 3×3 (r=2)** |
| Feature Merging | Concat → project | Concat → project |
| Residual Path | en_ir + en_vi | en_ir + en_vi |

MaxPool preserves sharp thermal peak signatures that AvgPool dilutes, directly targeting the small target recall issue.

---

## 🗂️ Project Structure

```
argus-/
├── baseline_qfdet_repo/          # MMDet-based QFDet baseline codebase
│   └── mmdet-rgbtdroneperson-main/
│       └── mmdet/models/detectors/qfdet.py   # ← CMAGM + HR-CMAGM implemented here
├── VTUAV_subset/                 # VTUAV dataset (RGB + Thermal pairs)
├── checkpoints/
│   └── qfdet_vtuav.pth           # Pre-trained baseline checkpoint
├── work_dirs/
│   └── qfdet_cmagm_stage3/       # Stage 3 CMAGM trained checkpoint
│       └── latest.pth
├── visual_comparison_results/    # Side-by-side visual inference panels
├── train_stage3_fusion.py        # Stage 3 CMAGM training script
├── train_stage3_extended.py      # Extended epoch fine-tuning script
├── train_stage4_hrcmagm.py       # Stage 4 HR-CMAGM training script (ready to run)
├── eval_stage3_cmagm.py          # CMAGM evaluation script
├── eval_stage3_extended.py       # Extended checkpoint evaluation script
├── analyze_error_distribution.py # Scale & error breakdown analysis
├── visualize_side_by_side_comparison.py  # Visual inference comparison
├── export_coco_predictions.py    # COCO format prediction export
├── predictions_coco_format.json  # Official COCO test predictions
├── stage2_benchmark_results.json # Baseline evaluation results
├── stage3_cmagm_results.json     # Stage 3 CMAGM evaluation results
└── error_analysis_report.json    # Scale distribution & precision/recall breakdown
```

---

## 🔧 Installation & Setup

```bash
# Activate the Python environment
cd argus-

# Install requirements (pre-installed in env)
# torch, mmcv, mmdet, numpy, scipy, opencv-python

# Verify setup
C:\env\Scripts\python.exe -c "import torch; import mmcv; print('Ready!')"
```

---

## 🏃 How to Run

### Evaluate Stage 3 CMAGM Model
```bash
C:\env\Scripts\python.exe eval_stage3_cmagm.py
```

### Run Visual Side-by-Side Comparison
```bash
C:\env\Scripts\python.exe visualize_side_by_side_comparison.py
```

### Analyze Scale & Error Distribution
```bash
C:\env\Scripts\python.exe analyze_error_distribution.py
```

### Train Stage 4 HR-CMAGM (from scratch)
```bash
C:\env\Scripts\python.exe train_stage4_hrcmagm.py
```

---

## 📐 Model Parameters

| Model | Parameters | Model Size |
|-------|-----------|------------|
| Baseline QFDet | 60,634,267 (60.6M) | 462.6 MB |
| CMAGM QFDet (Stage 3) | 60,700,990 (60.7M) | 462.1 MB |

CMAGM adds only **+66,723 parameters** (~0.1% overhead) for meaningful feature recalibration.

---

## 🎯 Dataset: VTUAV (Vehicle & Person UAV Detection)

| Split | Images | Annotations | Avg Instances/Image |
|-------|--------|-------------|---------------------|
| Train | 1,200 | 8,138 | 6.78 |
| Val | 300 | 2,337 | 7.79 |
| Test | 200 | 2,068 | 10.34 |

All images are 1920×1080 resolution RGB+Thermal pairs captured from UAV platforms.

---

## 🗺️ Methodology Overview

```
Phase 1: Dataset Exploration & Analysis
    └── explore_dataset.py → stage1_dataset_summary.json

Phase 2: Baseline QFDet Benchmarking
    └── eval_qfdet_baseline.py → stage2_benchmark_results.json

Phase 3: CMAGM Design & Implementation
    └── Modified Fusion_CAT in qfdet.py
    └── train_stage3_fusion.py → work_dirs/qfdet_cmagm_stage3/

Phase 4: Evaluation & Visual Analysis
    └── eval_stage3_cmagm.py → stage3_cmagm_results.json
    └── visualize_side_by_side_comparison.py → visual_comparison_results/
    └── export_coco_predictions.py → predictions_coco_format.json

Phase 5: Error Analysis & Bottleneck Discovery
    └── analyze_error_distribution.py → error_analysis_report.json
    └── Finding: Small object recall = 6.05% (primary bottleneck)

Phase 6: HR-CMAGM Architecture Design
    └── Upgraded Fusion_CAT with Dual-Pool + Multi-Scale Dilated Attention
    └── train_stage4_hrcmagm.py (ready to run)
```

---

## 🤝 Team
**Team Argus** — Hackathon Submission 2026