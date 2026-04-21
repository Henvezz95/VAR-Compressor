<div align="center">
  <h1>[ Infinity-VAR : DeepCompressor ]</h1>
  <p><em>8B Bitwise Autoregressive Generation on Edge GPUs</em></p>
</div>

<p align="center">
    <a href="https://github.com/mit-han-lab/deepcompressor/blob/master/LICENSE">
        <img alt="Apache License" src="https://img.shields.io/github/license/mit-han-lab/deepcompressor">
    </a>
</p>

## About This Fork

This repository is a specialized fork of the DeepCompressor framework, tailored specifically to democratize high-fidelity Visual Autoregressive (VAR) models for edge deployment. 

* **What this fork adds:** We introduce a comprehensive W4A4 and INT8 KV-cache quantization pipeline specifically designed for the **Infinity** family of generative models (2B and 8B). It mitigates extreme activation outliers using SVDQuant and compresses the monotonically growing KV-cache via Asymmetric Per-Channel INT8 Quantization. This allows the 8B model to run natively on 16GB edge silicon.
* **Paper:** For full methodological details, evaluation metrics, and edge hardware deployment strategies on NVIDIA Jetson architectures, please refer to our paper: [*Enabling 8B Bitwise Autoregressive Image Generation on Edge GPUs*](https://iris.unimore.it/handle/11380/1400428) (Available Soon).

## Acknowledgements & Upstream Projects

This work builds upon exceptional foundational research. For additional insights, upstream features, and the original codebases, please refer to the following projects:

* **SVDQuant & DeepCompressor:** The foundational quantization engine used in this fork. SVDQuant absorbs outliers by shifting them from activations to weights, then employing a high-precision low-rank branch with Singular Value Decomposition (SVD). For the original implementation, additional diffusion model support, and LLM quantization (QServe), visit the [MIT HAN Lab DeepCompressor repository](https://github.com/mit-han-lab/deepcompressor) and read the [SVDQuant paper](http://arxiv.org/abs/2411.05007).
* **Infinity VAR:** The target architecture of this fork. Infinity is a Bitwise Visual AutoRegressive Modeling framework capable of generating high-resolution, photorealistic images by predicting bitwise tokens across scales. It refactors visual generation with an infinite-vocabulary classifier and bitwise self-correction. For core model insights, visit the [Infinity Project Page & GitHub](https://foundationvision.github.io/infinity.project/) and read the [Infinity paper](https://arxiv.org/abs/2412.04431).

## Installation

### Install from Source

1. Clone this repository and navigate to the folder:
```bash
git clone https://github.com/Henvezz95/deepcompressor.git
cd deepcompressor
```

2. Install dependencies:
```bash
pip install -e .
cd Infinity
pip install -r requirements.txt
```
## Infinity VAR: 8B Bitwise Autoregressive Image Generation on Edge GPUs

Visual Autoregressive models achieve state-of-the-art fidelity, but the monotonically growing KV-cache introduces a severe Memory Wall, confining these systems to data-center infrastructure. This fork provides a specialized compression pipeline to break that wall.

Through structural profiling, we diagnosed extreme activation outliers in the FFN down-projections of the Infinity architecture (peaking at 353x the median). To resolve this, `ptq_infinity.py` extends the **SVDQuant** paradigm to VAR models, decoupling outliers via a low-rank branch. To mitigate the cache footprint without runtime overhead, we implement **Asymmetric Per-Channel INT8 Quantization**, mapping highly skewed channel variances to static 8-bit limits optimized via Golden-Section Search. 

This pipeline reduces the peak memory of the Infinity 8B model by 64% (from 37.1 GB to 13.3 GB), enabling local execution on mid-range edge devices. 

### Generative Quality Evaluation
Below is the generation quality evaluated with 5,000 samples from the MJHQ-30K dataset. Our quantization pipeline retains near-FP16 aesthetic alignment (ImageReward) while compressing the model severely.

| Model | Precision | Method | FID (↓) | ImageReward (↑) | CLIP-IQA (↑) |
|---|---|---|---|---|---|
| Infinity 8B | FP16 | -- | 19.6 | 1.18 | 0.945 |
| | INT W4A4 | SVDQuant + KV8 | 19.0 | 1.13 | 0.935 |
| Infinity 2B | FP16 | -- | 21.3 | 0.981 | 0.947 |
| | INT W4A4 | SVDQuant + KV8 | 20.2 | 0.840 | 0.919 |

### Hardware Efficiency Benchmarks
System footprint and end-to-end latency measured on an NVIDIA Jetson AGX Orin 64GB. The "Feasible HW" tier indicates the minimum commercial module required to run the model natively in memory.

| Model | Precision | Peak Memory | Latency | Feasible HW |
|---|---|---|---|---|
| Flux.1-dev | INT W4A4 | 11.8 GB | 112.0 s | Orin NX (16GB) |
| Infinity 8B | FP16 | 37.1 GB | 25.1 s | AGX Orin (64GB) |
| Infinity 8B | INT W4A4 + KV8 | **13.3 GB** | **27.0 s** | **Orin NX (16GB)** |
| Infinity 2B | FP16 | 16.0 GB | 8.46 s | AGX Orin (32GB) |
| Infinity 2B | INT W4A4 + KV8 | **7.71 GB** | **11.5 s** | **Orin Nano (8GB)** |

