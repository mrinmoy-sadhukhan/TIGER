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
      ┌─────┴───────────────┐
      │                     │
      ▼                     ▼
 Person ReID          Homography Projection
 Appearance                World Position
 Embeddings                    │
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

Example environment check:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

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

The existing ablation study evaluates TIGER under progressively stronger observation/oracle settings to quantify the influence of detector quality and local identity information.

### Table II. Ablation Study of the TIGER Framework on the WILDTRACK Dataset

| Method | IDF1 ↑ | MOTA ↑ | MOTP ↑ | MT ↑ | ML ↓ |
|---|---:|---:|---:|---:|---:|
| **YOLOX + BoT-SORT-ReID + TIGER** | 89.68 | 90.67 | 85.20 | 88.96 | 4.30 |
| **GT bbox + BoT-SORT-ReID + TIGER** | 93.68 | 98.63 | 92.10 | 96.10 | 3.90 |
| **GT bbox + GT label-ReID + TIGER** | **95.40** | **100.00** | **99.90** | **100.00** | **0.00** |

The oracle configurations show that improved detections and more reliable local identity information significantly strengthen global reassociation performance.

---

## Reviewer-Requested Component-Wise Ablation

To isolate the contribution of individual TIGER modules, the following leave-one-component-out ablation is recommended. This table should be populated only after the corresponding experiments are executed.

### Table III. Recommended Component-Wise Ablation of TIGER

| Configuration | IDF1 ↑ | MOTA ↑ | MOTP ↑ | MT ↑ | ML ↓ |
|---|---:|---:|---:|---:|---:|
| Full TIGER | 89.68 | 90.67 | 85.20 | 88.96 | 4.30 |
| TIGER without Tracklet Transformer | TBD | TBD | TBD | TBD | TBD |
| TIGER without Homography-Based Spatial Cue | TBD | TBD | TBD | TBD | TBD |
| TIGER without Motion Consistency | TBD | TBD | TBD | TBD | TBD |
| TIGER without Temporal Consistency | TBD | TBD | TBD | TBD | TBD |
| TIGER without Dynamic Gallery Update | TBD | TBD | TBD | TBD | TBD |
| TIGER without Emergency Relinking | TBD | TBD | TBD | TBD | TBD |

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

## Computational Complexity

The computational cost of the proposed reassociation stage is mainly determined by the Tracklet Transformer and query-gallery similarity computation.

For \(N\) tracklets of length \(L\), embedding dimension \(d\), \(M\) Transformer layers, \(Q\) query tracklets, and \(G\) gallery identities, the approximate computational complexity is:

\[
\mathcal{O}\left(MN\left(L^{2}d + Ld^{2}
ight) + QGd
ight).
\]

Homography projection, motion consistency, temporal consistency, and score fusion introduce comparatively minor or linear computational overhead.

---

## Hyperparameter Selection and Sensitivity

The TIGER framework uses appearance, spatial, motion, and temporal weighting factors together with association thresholds, temporal decay parameters, and a finite global gallery size. These hyperparameters were selected empirically during framework development and were kept fixed during the reported evaluation rather than being adjusted for individual test sequences.

The main parameters requiring sensitivity analysis include:

- Appearance similarity weight
- Spatial consistency weight
- Motion consistency weight
- Temporal consistency weight
- Association thresholds, including 0.58 and 0.60
- Exponential temporal-decay constants
- Global identity gallery size

A systematic sensitivity study is recommended for the extended evaluation. The following table can be populated after the corresponding experiments are completed.

### Recommended Hyperparameter Sensitivity Analysis

| Parameter | Tested Values | Selected Value | IDF1 ↑ | MOTA ↑ |
|---|---|---:|---:|---:|
| Association threshold | TBD | 0.58 / 0.60 | TBD | TBD |
| Appearance weight | TBD | TBD | TBD | TBD |
| Spatial weight | TBD | TBD | TBD | TBD |
| Motion weight | TBD | TBD | TBD | TBD |
| Temporal weight | TBD | TBD | TBD | TBD |
| Temporal decay constant | TBD | TBD | TBD | TBD |
| Gallery size | TBD | TBD | TBD | TBD |

No unmeasured sensitivity results are reported in this repository.

---

## Dataset Scope and Generalizability

The current evaluation is conducted on the **WILDTRACK** benchmark, which provides synchronized and calibrated multi-camera views and is well suited to evaluating the spatial and cross-camera identity reasoning used by TIGER.

Evaluation on additional multi-camera datasets such as **DukeMTMC**, **CityFlow**, or **Campus** would provide stronger evidence of generalizability across different camera layouts, scene densities, viewpoints, and environmental conditions. Such cross-dataset evaluation is therefore considered an important direction for an extended study.

---

## Statistical Significance and Result Stability

The current tables report the tracking metrics obtained from the evaluated TIGER configurations. Because some improvements over recent methods are relatively small, repeated experiments with different random seeds are recommended for the learned Tracklet Transformer component to quantify run-to-run variation.

A suitable reporting format is:

```text
IDF1 = mean ± standard deviation
MOTA = mean ± standard deviation
```

For example, three or more independent runs can be used to report the mean and standard deviation and, where appropriate, a 95% confidence interval. Statistical significance should only be claimed after these repeated experiments have been performed.

### Recommended Stability Table

| Run / Seed | IDF1 ↑ | MOTA ↑ | MOTP ↑ |
|---|---:|---:|---:|
| Seed 1 | TBD | TBD | TBD |
| Seed 2 | TBD | TBD | TBD |
| Seed 3 | TBD | TBD | TBD |
| **Mean ± SD** | **TBD** | **TBD** | **TBD** |

No statistical-significance values are fabricated or inferred from single-run results.


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

## Recommended Result Order

For both the paper and repository, the results are best presented in the following order:

1. **Table I — Comparison with Existing WILDTRACK Methods**  
   Establishes the overall performance of TIGER relative to previous approaches.

2. **Table II — Existing Oracle / Input-Quality Ablation**  
   Shows how detection and local identity quality affect final tracking performance.

3. **Table III — Component-Wise TIGER Ablation**  
   Directly isolates the contribution of the Tracklet Transformer, homography, motion, temporal consistency, gallery update, and emergency relinking.

4. **Computational Performance Table**  
   Reports FPS, latency, inference time, GPU memory, and RAM usage after tracking accuracy has been established.

This ordering first answers **“How well does TIGER perform?”**, then **“What affects its performance?”**, then **“Which TIGER components contribute?”**, and finally **“How expensive is the framework computationally?”**

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
  author  = {Author information to be added},
  journal = {Conference/Journal information to be added},
  year    = {2026}
}
```

---

## License

Add the appropriate license for the repository before public release.
