# MSSTNet (Submission Version)

This repository provides the official implementation of **MSSTNet**, 
a physics-aware multi-scale spatiotemporal network for robust motor imagery (MI) EEG decoding.

The code is released to support the reproducibility of experiments reported in the paper:
> *Physics-Aware Multi-Scale Spatiotemporal Representation Learning for Robust Motor Imagery EEG Decoding*

*(Author information has been omitted for double-blind review)*

---

## 1. Overview

Decoding motor imagery-based EEG (MI-EEG) remains challenging due to inherent spatiotemporal non-stationarity, such as volume conduction, source mixing, and latency jitter. To address these issues, **MSSTNet** introduces:
- **PAMSE (Physics-Aware Multi-Scale Spatial Encoding):** Leverages 3D electrode coordinates to construct distance-consistent neighborhoods for robust spatial feature aggregation.
- **AMTP (Adaptive Multi-Scale Temporal Pyramid):** Integrates RMS-based adaptive pooling and a random ratio cropping strategy to extract multi-resolution temporal features and improve tolerance to temporal misalignment.
- **CSCA (Cross-Scale Contrastive Alignment):** Bridges the semantic gap between scales, enforcing representation consistency while preserving discriminability.

The current release focuses on reproducing the experimental results on the **BCI Competition IV-2a** dataset. All model parameters have been pre-configured to their default optimal settings as reported in the manuscript, allowing for straightforward reproduction.

---

## 2. Code Status 

- The repository contains all necessary files to **run and reproduce the reported results** on the BCI-IV-2a dataset.
- The code is **executable as provided**, but has not yet been fully refactored into a modular library.
- Only the **BCI-IV-2a** experimental pipeline and the core MSSTNet architecture are included in this initial submission version.
- Additional datasets (e.g., OpenBMI), comprehensive configurations, and complete code cleanup will be released publicly after the peer-review process is concluded.

---

## 3. Requirements

The implementation is based on Python and PyTorch. 

Recommended environment:
- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy, SciPy, scikit-learn

---
