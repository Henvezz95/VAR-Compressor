<h1 align="center"><code><b>[ Infinity-VAR : DeepCompressor ]</b></code></h1>

<p align="center"><b><i>8B Bitwise Autoregressive Generation on Edge GPUs</i></b></p>

<p align="center">
    <a href="https://github.com/mit-han-lab/deepcompressor/blob/master/LICENSE">
        <img alt="Apache License" src="https://img.shields.io/github/license/mit-han-lab/deepcompressor">
    </a>
</p>

# About This Fork

This repository is a specialized fork of the DeepCompressor framework, tailored specifically to democratize high-fidelity Visual Autoregressive (VAR) models for edge deployment. 

* **What this fork adds:** We introduce a comprehensive W4A4 and INT8 KV-cache quantization pipeline specifically designed for the **Infinity** family of generative models (2B and 8B). It mitigates extreme activation outliers using SVDQuant and compresses the monotonically growing KV-cache via Asymmetric Per-Channel INT8 Quantization. This allows the 8B model to run natively on 16GB edge silicon.
* **Paper:** For full methodological details, evaluation metrics, and edge hardware deployment strategies on NVIDIA Jetson architectures, please refer to our paper: [*Enabling 8B Bitwise Autoregressive Image Generation on Edge GPUs*](https://iris.unimore.it/handle/11380/1400428) (Available Soon).

# Acknowledgements & Upstream Projects

This work builds upon exceptional foundational research. For additional insights, upstream features, and the original codebases, please refer to the following projects:

* **SVDQuant & DeepCompressor:** The foundational quantization engine used in this fork. SVDQuant absorbs outliers by shifting them from activations to weights, then employing a high-precision low-rank branch with Singular Value Decomposition (SVD). For the original implementation, additional diffusion model support, and LLM quantization (QServe), visit the [MIT HAN Lab DeepCompressor repository](https://github.com/mit-han-lab/deepcompressor) and read the [SVDQuant paper](http://arxiv.org/abs/2411.05007).
* **Infinity VAR:** The target architecture of this fork. Infinity is a Bitwise Visual AutoRegressive Modeling framework capable of generating high-resolution, photorealistic images by predicting bitwise tokens across scales. It refactors visual generation with an infinite-vocabulary classifier and bitwise self-correction. For core model insights, visit the [Infinity Project Page & GitHub](https://foundationvision.github.io/infinity.project/) and read the [Infinity paper](https://arxiv.org/abs/2412.04431).

# Installation

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

# Usage

To quantize the Infinity models, execute the `ptq_infinity.py` script. The framework relies on a strict configuration hierarchy. You must pass the YAML files as positional arguments to construct the pipeline: defining the model, setting the calibration parameters, and dictating the specific quantization math.

# 0. Technical Motivation: Diagnostic Profiling

The quantization strategies implemented in this repository are driven by a deep structural analysis of the Infinity VAR architecture. Unlike standard LLMs, Visual Autoregressive models exhibit unique activation patterns that necessitate specialized treatment.

To reproduce our diagnostic findings, use the profiling suite:
```bash
python -m evaluation.activations_measurements configs/models/infinity-8b.yaml configs/collect/qdiff.yaml
```

### The Outlier Problem (Linear Layers)
Through our profiling, we identified extreme activation outliers in the **FFN down-projections**, with Kurtosis values significantly exceeding Gaussian distributions.

* **Max-to-Median Ratio**: Reaches up to **353x** in the 8B model.
* **Implication**: Standard Min-Max quantization would lead to massive precision loss; this justifies our use of **SVDQuant** to decouple these outliers into a high-precision low-rank branch.

### KV-Cache Variance (Self-Attention)
Analysis of the monotonically growing KV-cache reveals that variance is not uniform across dimensions.

| Metric | Measured Value (8B) | Technical Requirement |
| :--- | :--- | :--- |
| **$CV_{channel}$** | > 1.2 | **Per-Channel Scaling**: Variance is driven by specific channels rather than tokens. |
| **Skewness** | ~0.85 (Key Cache) | **Asymmetric Mapping**: Distributions are highly skewed, requiring non-centered zero-points. |

By running the diagnostic script, users can verify that these structural characteristics are consistent across both 2B and 8B variants, validating the selection of **Asymmetric Per-Channel INT8** for the cache pipeline.

## 1. Running Quantization

The following command executes the baseline quantization pipeline for the Infinity 8B model, utilizing the specific calibration settings defined in `qdiff.yaml` and running the complete **INT4 SVDQuant** pipeline (incorporating both activation smoothing and low-rank weight branches):

```python
python -m deepcompressor.app.diffusion.ptq_infinity configs/models/infinity-8b.yaml configs/collect/qdiff.yaml configs/svdquant/int4.yaml
```

**Configuration Override Hierarchy:** Positional arguments dictate the override order (files passed later in the command override overlapping keys in earlier files). Ensure all files remain within the relative paths of your working directory. For rapid debugging, you can append additional override configurations (e.g., reducing calibration steps via `fast.yaml`) at the end of the execution chain:

```python
python -m deepcompressor.app.diffusion.ptq_infinity configs/models/infinity-8b.yaml configs/collect/qdiff.yaml configs/svdquant/int4.yaml configs/svdquant/fast.yaml
```

*(Note: End-to-end evaluation and image generation using the quantized models are handled via separate evaluation scripts, not during this initial PTQ pass).*

### Example Configurations

The repository provides several example configurations to demonstrate different quantization strategies for the Infinity models.

* **Base Model Configurations:**
    * `configs/models/infinity-8b.yaml`: Defines the pipeline architecture, precision (W4A4 + SVDQuant LoRA), and paths for the 8B model.
    * `configs/models/infinity-2b.yaml`: Defines the pipeline architecture, precision (W4A4 + SVDQuant LoRA), and paths for the 2B model.

* **Quantization Strategies:**
    * `configs/models/infinity-2b-smoothquant.yaml`: Enables activation smoothing to mitigate outliers without utilizing the low-rank branch for weights.
    * `configs/models/infinity-2b-naive.yaml`: Performs standard block-wise quantization (e.g., 64-group) on the weights. This is useful as a baseline but may cause degradation, especially in the 2B model.

## 2. KV-Cache Calibration

To generate the optimal Asymmetric Per-Channel INT8 quantization scales for the KV-cache, execute the `calibrate_cache_quantization` module. Unlike standard LLM cache quantization, our analysis of VAR models indicates that variance is predominantly channel-driven across both Keys and Values. 

The script employs a **Golden-Section Search** to optimize clipping bounds per channel, minimizing the reconstruction Mean Squared Error (MSE). This deterministic strategy accommodates highly skewed distributions (peaking at 11.56 in the 2B Key Cache) without the control-flow overhead of dynamic token pruning.

It requires the base model configuration and the calibration collection parameters:

```bash
python -m deepcompressor.app.diffusion.calibrate_cache_quantization configs/models/infinity-8b.yaml configs/collect/qdiff.yaml
```

**Key Implementation Details:**
* **Asymmetric Mapping:** Uses affine quantization to align scaling factors with the axes of highest variance and shift zero-points to accommodate skewed dynamic ranges.
* **Optimization:** Scans a logarithmic grid of percentiles before refining the optimal clipping bounds using a coarse-to-fine search.
* **Output:** Generates the `scale` and `zero_point` parameters saved to `kv_scales/kv_quant_calib.pt`, which are required to run the full W4A4+KV8 inference pipeline.

*(Note: This routine calculates the `scale` and `zero_point` parameters saved to `kv_scales/kv_quant_calib.pt`, which are subsequently required to run the full W4A4+KV8 inference pipeline).*

## 3. Quality Evaluation (Fake-Quantization)

To assess the generative fidelity (FID, ImageReward, CLIP-IQA) before deploying to edge hardware, the `benchmark_assembled_model.py` script provides a bit-accurate simulation of the quantization noise. By using **fake-quantization**, the framework applies low-bit logic (e.g., INT4 or INT8) to the model weights and activations while performing the underlying computation in `bfloat16`.

This allows for granular ablation studies—independently toggling Weight, Activation, and KV-cache quantization to identify the precise impact on aesthetic quality.

### Running the Evaluation

The following command evaluates an Infinity 8B model on the **MJHQ** and **DCI** benchmarks. It simulates a complete **W4A4 + KV8** pipeline by fusing the SVD low-rank branches and activation scales generated during the PTQ phase:

```bash
python -m evaluation.benchmark_assembled_model \
    configs/models/infinity-8b.yaml \
    configs/svdquant/int4.yaml \
    --ref-root ./evaluation_output/infinity_fp16_8b \
    --gen-root ./evaluation_output/infinity_w4a4_kv8_8b \
    --base-path ./runs/diffusion/int4_rank32_8b/ \
    --enable_weight_quant true \
    --enable_activation_quant true \
    --enable_kv_quant true \
    --eval-benchmarks MJHQ DCI \
    --eval-num-samples 5000 \
    --eval-gt-metrics clip_iqa clip_score fid image_reward psnr ssim lpips
```

### Script Arguments

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--base-path` | `str` | Directory containing the PTQ artifacts: `model.pt`, `smooth.pt`, and `branch.pt`. |
| `--enable_weight_quant` | `bool` | Enables fake-quantization for transformer weights. |
| `--enable_activation_quant`| `bool` | Enables fake-quantization for linear layer input activations. |
| `--enable_kv_quant` | `bool` | Enables Asymmetric Per-Channel INT8 simulation for the KV-cache using optimized scales. |
| `--gen-root` | `str` | Destination for generated images and the final `results.json`. |
| `--ref-root` | `str` | Path to the ground-truth reference dataset for metrics that require a reference (e.g., SSIM, PSNR, LPIPS). |

**Note on Artifacts:** The script automatically looks for cache scales in `runs/kv_scales/kv_quant_calib.pt`. Ensure you have run the `calibrate_cache_quantization` script before enabling the `--enable_kv_quant` flag.

## 4. Performance & Memory Benchmarking (Real Quantization)

To measure the actual memory savings and inference speed on edge hardware (e.g., NVIDIA Jetson), use the `infinity_w4a4_test.py` script. Unlike the quality evaluation script, this routine swaps standard layers for real **SVDQuantLinear** modules and executes optimized low-bit kernels.

### Prerequisites

This script requires specialized hardware-accelerated kernels for 4-bit weight and 4-bit activation computation. You must install the following dependency:

* **Nunchaku (Specialized Fork):** [Henvezz95/nunchaku-fork](https://github.com/Henvezz95/nunchaku-fork)

### Benchmarking Workflow

The script performs the following hardware validation steps:
1. **Model Transformation:** Swaps standard `nn.Linear` layers for `SVDQuantLinear` (defaulting to Rank-32) while excluding sensitive layers like the transformer head and embeddings.
2. **Artifact Injection:** Loads the real quantized weights (`model.pt`) and the high-precision SVD branches (`branch.pt`) directly into the specialized modules.
3. **Cache Activation:** Integrates the calibrated INT8 KV-cache parameters via the `attach_kv_qparams` utility.
4. **Footprint Profiling:** Executes multiple generation loops and reports the absolute peak GPU memory usage using `torch.cuda.max_memory_allocated()`.

### Usage

The real-quantization benchmark now follows the same configuration hierarchy as the rest of the **VAR-Compressor** pipeline. You must provide the model YAML file and the path to your PTQ artifacts:

```bash
# Benchmark the 8B model with real W4A4 + KV8 kernels
python infinity_w4a4_test.py configs/models/infinity-8b.yaml \
    --base-path ./runs/diffusion/int4_rank32_8b/ \
    --enable_kv_quant true \
    --prompt "A cinematic photo of a robot in Zurich"
```

**Custom Arguments:**
* `--base-path`: (Required) The directory containing your `model.pt`, `smooth.pt`, and `branch.pt` artifacts.
* `--enable_kv_quant`: Set to `true` to enable real INT8 KV-cache kernels.
* `--prompt`: The text description used for the generation benchmark.
* `--seed`: Fixed seed for reproducibility during latency measurement.

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
