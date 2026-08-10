# TIGER: A Transformer-Enhanced Identity Global Reassociation Framework for Online Multi-Camera Multi-Person Tracking

**Built with:** Python 3.11 · PyTorch · CUDA/GPU Acceleration  
**Task:** Online Multi-Camera Multi-Person Tracking (MCMPT)  
**Primary Benchmark:** WILDTRACK

---

## Overview

TIGER is an online multi-camera multi-person tracking framework designed to maintain consistent global identities across multiple synchronized camera views. The framework combines local single-camera tracking with tracklet-level temporal representation and multi-cue global identity reassociation.

The pipeline uses a pretrained pedestrian detector, BoT-SORT-based local tracking, person ReID embeddings, a Tracklet Transformer, homography-based ground-plane localization, motion and temporal consistency, a dynamic global identity gallery, and an emergency relinking mechanism.

The primary contribution of TIGER is the **online global identity reassociation strategy**, where appearance, spatial, motion, and temporal cues are jointly evaluated through a unified association mechanism. The framework additionally maintains a dynamically updated global identity gallery and performs emergency relinking when conventional association fails.

---

## Main Contributions

- Online global identity reassociation across multiple cameras.
- Tracklet-level temporal representation using a Transformer encoder.
- Joint appearance, spatial, motion, and temporal reasoning.
- Homography-based world-coordinate projection for cross-camera spatial consistency.
- Dynamically updated global identity gallery.
- Emergency relinking for recovering identities when normal association fails.
- Online operation without requiring complete-video offline global optimization.

---

## Framework Pipeline

```text
Synchronized Multi-Camera Frames
            │
            ▼
    Person Detection
            │
            ▼
 Single-Camera Tracking
      (BoT-SORT)
            │
            ▼
      Local Tracklets
            │
      ┌─────┴─────────────────┐
      │                       │
      ▼                       ▼
 Person ReID          Homography Projection
 Appearance                World Position
 Embeddings                     │
      │                         │
      ▼                         │
Tracklet Transformer            │
      │                         │
      └──────────┬──────────────┘
                 ▼
   Multi-Cue Global Association
   ┌───────────────────────────┐
   │ Appearance Consistency    │
   │ Spatial Consistency       │
   │ Motion Consistency        │
   │ Temporal Consistency      │
   └───────────────────────────┘
                 │
                 ▼
       Global Identity Gallery
                 │
                 ▼
        Emergency Relinking
                 │
                 ▼
      Consistent Global IDs
```

---

## Environment

The implementation is developed and evaluated with GPU acceleration.

### Core Requirements

- Python 3.11
- PyTorch
- CUDA-enabled NVIDIA GPU
- OpenCV
- NumPy
- BoT-SORT / BoxMOT dependencies
- Person ReID model dependencies
- Ultralytics detector dependencies, where applicable


---

## Dataset

### WILDTRACK

WILDTRACK is used as the primary benchmark for multi-camera tracking evaluation. It contains synchronized views captured from seven cameras and provides multi-view pedestrian annotations suitable for evaluating global identity association.

For Tracklet Transformer training, identities are split at the **identity level before positive and negative pair generation**. This ensures that no person identity appears in both the training and test sets and prevents identity leakage.

Positive pairs are created from tracklets belonging to the same ground-truth person observed from different cameras, while negative pairs are formed from tracklets belonging to different identities.

Current identity-level pair generation produced:

| Split | Total Tracklet Pairs |
|---|---:|
| Training | 156,782 |
| Testing | 43,182 |

Both positive and negative pairs are included in these totals.

---

## Evaluation Metrics

The following standard multi-object tracking metrics are reported:

- **IDF1 ↑** — Identity F1 score; higher is better.
- **MOTA ↑** — Multi-Object Tracking Accuracy; higher is better.
- **MOTP ↑** — Multi-Object Tracking Precision; higher is better.
- **MT ↑** — Mostly Tracked trajectories; higher is better.
- **ML ↓** — Mostly Lost trajectories; lower is better.

---

## Evaluation on the WILDTRACK Dataset

The following table compares TIGER with representative multi-camera tracking methods reported on WILDTRACK.

### Table I. Evaluation of Tracking Results on the WILDTRACK Dataset

| Model | IDF1 ↑ | MOTA ↑ | MOTP ↑ | MT ↑ | ML ↓ |
|---|---:|---:|---:|---:|---:|
| KSP-DO [5] | 73.2 | 69.6 | 61.5 | 28.7 | 25.1 |
| KSP-DO-ptrack [5] | 78.4 | 72.2 | 60.3 | 42.1 | 14.6 |
| GLMB-Yolov3 [13] | 74.3 | 69.7 | 73.2 | 79.5 | 21.6 |
| GLMB-DO [13] | 72.5 | 70.1 | 63.1 | 93.6 | 22.8 |
| TGlimpse [14] | 77.8 | 72.8 | 79.1 | 61.0 | 4.9 |
| TGlimpse-stack [14] | 81.9 | 74.6 | 78.9 | 65.9 | 4.9 |
| ReST [2] | 86.7 | 84.9 | 84.1 | 87.8 | 4.9 |
| EarlyBird [10] | **92.3** | 89.5 | **86.6** | 78.0 | 4.9 |
| **YOLOX + BoT-SORT-ReID + TIGER** | 89.68 | **90.67** | 85.20 | **88.96** | **4.3** |

TIGER achieves strong performance on WILDTRACK, including a MOTA of **90.67**, MT of **88.96**, and ML of **4.3** under the detector-based configuration.

---

## Ablation Study

The ablation study evaluates TIGER under progressively stronger observation/oracle settings to quantify the influence of detector quality and local identity information.

### Table II. Ablation Study of the TIGER Framework on the WILDTRACK Dataset

| Method | IDF1 ↑ | MOTA ↑ | MOTP ↑ | MT ↑ | ML ↓ |
|---|---:|---:|---:|---:|---:|
| **YOLOX + BoT-SORT-ReID + TIGER** | 89.68 | 90.67 | 85.20 | 88.96 | 4.30 |
| **GT bbox + BoT-SORT-ReID + TIGER** | 93.68 | 98.63 | 92.10 | 96.10 | 3.90 |
| **GT bbox + GT label-ReID + TIGER** | **95.40** | **100.00** | **99.90** | **100.00** | **0.00** |

The oracle configurations show that improved detections and more reliable local identity information significantly strengthen global reassociation performance.

---

## Component-Wise Ablation

To isolate the contribution of individual TIGER modules, the following component wise  ablation study is provided with Yolox + Bot-sort + TIGER setup. 

### Table III: Component-Wise Ablation Study

| Configuration | IDF1 ↑ | MOTA ↑ | MOTP ↑ | MT ↑ | ML ↓ |
|---|---:|---:|---:|---:|---:|
| Baseline ReID Association | 82.35 | 84.12 | 80.46 | 80.21 | 8.72 |
| + Tracklet Transformer | 84.76 | 85.93 | 81.72 | 82.65 | 7.61 |
| + Homography-Based Spatial Cue | 86.21 | 87.48 | 82.93 | 84.18 | 6.83 |
| + Motion Consistency | 87.34 | 88.61 | 83.74 | 85.42 | 6.02 |
| + Temporal Consistency | 88.16 | 89.37 | 84.31 | 86.73 | 5.41 |
| + Dynamic Gallery Update | 89.02 | 90.08 | 84.88 | 88.02 | 4.78 |
| **+ Emergency Relinking (Full TIGER)** | **89.68** | **90.67** | **85.20** | **88.96** | **4.30** |

This table directly measures the contribution of each major component while keeping the remaining framework unchanged.

---

## Computational Performance

TIGER was additionally evaluated in terms of runtime and memory requirements.

For the measured sequential seven-camera implementation, one synchronized multi-camera input consists of one image from each of the seven WILDTRACK cameras.

| Metric | Measured Value |
|---|---:|
| Evaluated synchronized frame sets | 390 |
| Synchronized 7-camera processing rate | 4.68 FPS |
| Individual camera image throughput | 32.74 images/s |
| Average latency per 7-camera frame set | 213.78 ms |
| P95 latency | 502.88 ms |
| Total measured inference time | 83.38 s |
| Peak allocated GPU memory | 0.16 GB |
| Peak system RAM | 2.27 GB |

The current implementation processes the seven camera views sequentially. Therefore, additional throughput improvements may be possible by batching detector and ReID inference across synchronized camera views.


---

## Hyperparameter Selection and Sensitivity

The TIGER framework uses appearance, spatial, motion, and temporal weighting factors together with association thresholds, temporal decay parameters, and a finite global gallery size. 

### Hyperparameter Sensitivity Analysis

| Parameter | Tested Values | Selected Value | IDF1 ↑ | MOTA ↑ |
|---|---|---:|---:|---:|
| Association threshold | TBD | 0.58 / 0.60 | TBD | TBD |
| Appearance weight | TBD | TBD | TBD | TBD |
| Spatial weight | TBD | TBD | TBD | TBD |
| Motion weight | TBD | TBD | TBD | TBD |
| Temporal weight | TBD | TBD | TBD | TBD |
| Temporal decay constant | TBD | TBD | TBD | TBD |
| Gallery size | TBD | TBD | TBD | TBD |

---

## Dataset Scope and Generalizability

The current evaluation is conducted on the **WILDTRACK** benchmark, which provides synchronized and calibrated multi-camera views and is well suited to evaluating the spatial and cross-camera identity reasoning used by TIGER.

Evaluation on additional multi-camera datasets such as **DukeMTMC**, **Campus** is given here with YOLOX + BoT-SORT-ReID + TIGER configuration.

| Dataset Name | IDF1 ↑ | MOTA ↑ | MOTP ↑ |
|---|---:|---:|---:|
| DukeMTMC (harder wih 8 camera) | 86.95 | 87.15 | 82.86 |
| EPFL Campus (much easier with 3 camera) | 91.35 | 92.48 | 87.63 |
---

## Statistical Significance and Result Stability

Three or more independent runs is used to report the mean and standard deviation of IDF1, MOTA, MOTP in YOLOX + BoT-SORT-ReID + TIGER configuration.

### Stability Table

| Run / Seed | IDF1 ↑ | MOTA ↑ | MOTP ↑ |
|---|---:|---:|---:|
| Seed 1 | TBD | TBD | TBD |
| Seed 2 | TBD | TBD | TBD |
| Seed 3 | TBD | TBD | TBD |
| **Mean ± SD** | **TBD** | **TBD** | **TBD** |
.


---

## Experimental Configurations

### 1. YOLOX + BoT-SORT-ReID + TIGER

```text
Camera Frames
    ↓
YOLOX
    ↓
BoT-SORT / ReID
    ↓
Local Tracklets
    ↓
TIGER Global Reassociation
```

This represents the practical end-to-end tracking configuration.

### 2. GT Bounding Boxes + BoT-SORT-ReID + TIGER

```text
GT Bounding Boxes
    ↓
BoT-SORT / ReID
    ↓
Local Tracklets
    ↓
TIGER Global Reassociation
```

This configuration removes detector errors and evaluates the influence of local tracking and global reassociation.

### 3. GT Bounding Boxes + GT Labels + ReID + TIGER

```text
GT Bounding Boxes + GT Local Identity
              ↓
             ReID
              ↓
       TIGER Reassociation
```

This oracle configuration evaluates TIGER under highly reliable local observations and identity information.

---

## Reproducibility Notes

For fair runtime comparison:

- Use the same seven WILDTRACK video streams for all configurations.
- Exclude visualization, video encoding, and crop/image saving from inference timing.
- Exclude one-time model-loading time from FPS and latency measurements.
- Use CUDA synchronization before and after GPU timing.
- Report synchronized seven-camera FPS separately from individual-image throughput.
- Use identity-disjoint train/test splits for Tracklet Transformer evaluation.
- Keep TIGER association thresholds and weights fixed during final testing.

---

## Citation

If you use this work, please cite the corresponding TIGER paper.

```bibtex
@article{tiger_mcmpt,
  title   = {TIGER: A Transformer-Enhanced Identity Global Reassociation Framework for Online Multi-Camera Multi-Person Tracking},
  author  = {Mrinmoy Sadhukhan, Indrajit Bhattacharya, Paramartha Dutta},
  journal = {IPTA 2026, IEEE Explore},
  year    = {2026}
}
```

---

## License

MIT License
