# Wavelet Domain Adaptation

# Wavelet Domain Adaptation for Cross-Regional Landslide Segmentation

Code for the paper **"Beyond Pixels: Multi-Scale Wavelet Transfer for Cross-Regional Landslide Segmentation"**, accepted at IGARSS 2026.

> **Note:** This repository contains the experimental benchmarking code used to produce the paper's results. It is not a packaged library — see [Usage](#usage) for how to run each experiment.

---

## Overview

Deep learning models for landslide segmentation generalize poorly across geographic regions due to domain shift in terrain morphology, vegetation, and imaging conditions. We propose **Wavelet Domain Adaptation (WDA)**, a lightweight unsupervised method that blends target low-frequency approximation components (LL subband) into source images via a single discrete wavelet transform pass, while preserving source high-frequency structure (edges and spatial detail).

Across 240 source-target region pairs from the CAS Landslide Dataset, WDA improves mean IoU from 16.6% → 19.9% over direct transfer, and reduces large adaptation failures from 39 → 13 compared to Fourier Domain Adaptation (FDA) at α=0.40.

---

## Repository Structure

```
├── data/               # Place dataset contents here (see Dataset section)
├── figures/            # Generated figures (PDFs)
├── logs/               # Experiment logs (auto-created at runtime)
├── results/
│   ├── baseline/       # Direct transfer results
│   ├── edge_analysis/  # Edge distortion metrics across β values
│   ├── fourier/        # FDA results across β values
│   ├── similarity/     # Pairwise source-target similarity metrics
│   ├── wavelet/        # WDA results across α values
│   └── key_results_summary.txt
└── src/
    ├── config.py           # Dataset paths and hyperparameters
    ├── dataset.py          # CAS Landslide Dataset loader
    ├── model.py            # U-Net with ResNet-18 encoder
    ├── train.py            # Training loop
    ├── main.py             # Experiment entry point
    ├── wavelet.py          # WDA implementation
    ├── fourier.py          # FDA implementation
    ├── edge_analysis.py    # Edge distortion metrics
    ├── utils.py            # Shared utilities
    └── generate_figures.ipynb  # Reproduces all paper figures from results CSVs
```

---

## Dataset

This project uses the **CAS Landslide Dataset**, a large-scale multi-sensor benchmark covering 9 geographically distinct regions (16 sub-datasets).

**Citation:**
> Xu, Y., Ouyang, C., Xu, Q., Wang, D., Zhao, B., & Luo, Y. (2024). CAS landslide dataset: A large-scale and multisensor dataset for deep learning-based landslide detection. *Scientific Data*, 11(1), 12.

**Download:** [https://zenodo.org/records/10294997](https://zenodo.org/records/10294997)

**Setup:** Download and extract the archive, then copy all contents into the `data/` directory:

```
data/
├── Hokkaido/
├── Jiuzhai/
├── Lombok/
└── ...
```

The expected structure within each region is handled by `dataset.py`. See `config.py` to adjust paths if needed.

---

## Installation

```bash
git clone https://github.com/Rahuldeb5/wavelet-domain-adaptation
cd wavelet-domain-adaptation
```

Requires Python 3.8+. Tested on CUDA 12.8. Install PyTorch first for your CUDA version, then the remaining dependencies:

```bash
# Step 1: Install PyTorch (adjust cu121 to match your CUDA version)
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage

All experiments are run via `main.py` from the `src/` directory:

```bash
cd src
```

**Direct Transfer (baseline):**
```bash
python main.py --method baseline
```

**Wavelet Domain Adaptation:**
```bash
python main.py --method wavelet --alpha 0.40
```

**Fourier Domain Adaptation:**
```bash
python main.py --method fourier --beta 0.01
```

**Pairwise Source-Target Similarity:**
```bash
python main.py --method similarity
```

**Edge Distortion Analysis:**
```bash
python main.py --method edge
```

Results are saved to `results/` and logs to `logs/`. To reproduce all paper figures from existing CSVs, open `src/generate_figures.ipynb`.

---

## Results

Performance across all 240 source-target pairs:

| Method | IoU (%) | ΔIoU (%) | Pairs Improved | Large Drops (≤−10%) | Severe Drops (≤−20%) |
|---|---|---|---|---|---|
| Direct Transfer | 16.6 | — | — | — | — |
| FDA (β=0.01) | 16.4 | −0.2 | 48.3% | 39 | 14 |
| WDA (α=0.10) | 17.5 | +0.9 | 54.6% | 15 | 2 |
| WDA (α=0.20) | 18.1 | +1.5 | 60.0% | 8 | 2 |
| WDA (α=0.40) | **19.9** | **+3.3** | 59.6% | 13 | 2 |
| Oracle WDA | 25.6 | +9.0 | 90.4% | 0 | 0 |

---

## Citation

Paper DOI and full citation will be added after the conference proceedings are published.

---

## Acknowledgments

This work was supported in part by the U.S. National Science Foundation under Award No. 2425802, and the U.S. Department of Energy RENEW Award DE-SC0025738.
