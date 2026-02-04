# Opportunistic Promptable Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://www.arxiv.org/abs/2602.00309)
[![Zenodo](https://img.shields.io/badge/Zenodo-Model%20Weights-blue.svg)](https://zenodo.org/records/18394860)

Official implementation of **Opportunistic Promptable Segmentation** for 3D CT lesion segmentation using routine radiological annotations.

---

## Overview

This repository contains code and model weights for our paper:

> **Opportunistic Promptable Segmentation: Leveraging Routine Radiological Annotations to Guide 3D CT Lesion Segmentation**
> *Church et al.*

The method leverages existing clinical annotations (e.g., GSPS) as prompts for large-scale dataset generation.

---

## Method Overview

<p align="center">
  <img src="figures/method_overview.png" width="85%">
</p>

**Figure:** Overview of the opportunistic promptable segmentation pipeline. Routine radiological annotations are converted into prompts that guide a 3D segmentation model to produce lesion masks.

---

## Model Weights

Pretrained model weights are available on **Zenodo**:

👉 [https://zenodo.org/records/18394860]

## Paper

If you find this work useful, please consider citing:

```bibtex
@article{church2026sam2ct,
  title={Opportunistic Promptable Segmentation: Using Routine Radiological Annotations to Guide 3D CT Lesion Segmentation},
  author={Church, et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
