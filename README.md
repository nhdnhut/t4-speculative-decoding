# T4Decode - LLM Speculative Decoding and Quantization Suite

> A reproducible experimental framework for accelerating Large Language Model decoding on consumer and cloud GPU tiers (Google Colab NVIDIA T4).
> PyTorch | Hugging Face Transformers | Speculative Draft Decoding | bitsandbytes INT8 & NF4 | LoRA Distillation

[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

---

## 1. Overview

T4Decode evaluates the intersection of speculative draft decoding, parameter quantization (FP16, INT8, and NF4), and draft model distillation on constrained 16 GB NVIDIA T4 GPU instances.

Core Capabilities:
- Speculative Decoding Engine: Pairs a 3B parameter target model (`Qwen/Qwen2.5-3B-Instruct`) with an aligned 0.5B parameter draft model (`Qwen/Qwen2.5-0.5B-Instruct`), generating multiple candidate tokens per verification forward pass.
- Quantization Trade-off Analysis: Evaluates inference speed and token generation fidelity across full precision (FP16), bitsandbytes INT8, and 4-bit NormalFloat (NF4) double quantization.
- Draft Model LoRA Distillation: Fine-tunes the draft model directly on teacher verification trajectories using Low-Rank Adaptation (LoRA), increasing acceptance rates during speculation.
- VRAM Budget-Aware Configuration Selector: Implements an automated Pareto frontier selector that enforces peak VRAM constraints (<= 12 GiB) while guaranteeing accuracy loss relative to baseline does not exceed 3%.

---

## 2. Experimental Architecture

```
[Prompt Input]
      |
      v
[Draft Model (Qwen2.5-0.5B / LoRA)] ---> Generates K candidate tokens (speculative steps)
      |
      v
[Target Model (Qwen2.5-3B / NF4 or INT8)] ---> Parallel verification forward pass
      |
      +---> Accepted tokens: Emitted immediately
      +---> First rejected token: Resampled from target distribution, resetting draft context
```

Hardware and Hyperparameter Specification:
- Hardware Target: Single NVIDIA T4 GPU (16 GB nominal VRAM).
- Target Model: `Qwen/Qwen2.5-3B-Instruct` (locked by commit SHA).
- Draft Model: `Qwen/Qwen2.5-0.5B-Instruct` (FP16 / LoRA distilled).
- Attention Mechanism: Scaled Dot-Product Attention (SDPA) with dynamic KV-caching.
- Batch Size: 1 (interactive streaming regime).
- Maximum Context Window: 4096 tokens (prompt + completion).

---

## 3. Workflow Notebooks

The repository provides three modular, reproducible Jupyter notebooks designed for execution on Google Colab or local GPU workstations:

1. `notebooks/01_Baseline_T4.ipynb`: Environment verification, model SHA locking, baseline FP16 profiling, KV-cache validation, and metric logging.
2. `notebooks/02_Train_Optimize_T4.ipynb`: Teacher trajectory collection, draft model LoRA distillation, merged checkpoint export, and quantization sweep (FP16, INT8, NF4).
3. `notebooks/03_Evaluate_Demo_T4.ipynb`: End-to-end evaluation on held-out test splits, tokens-per-second throughput benchmarking, and interactive inference demo.

---

## 4. Quick Start Guide

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (or Google Colab T4 runtime)

### Installation
```bash
# Clone the repository
git clone https://github.com/nhutnhd/t4-speculative-decoding.git
cd t4-speculative-decoding

# Install dependencies
pip install -r requirements-colab.txt
```

### Running Unit Tests
```bash
pytest -c pytest.ini tests/test_core.py -v
```

### Building Packaged Notebooks
```bash
python build_notebooks.py
```

---

## 5. Directory Structure

```
t4-speculative-decoding/
├── notebooks/                  # Reproducible Colab notebooks
│   ├── 01_Baseline_T4.ipynb
│   ├── 02_Train_Optimize_T4.ipynb
│   └── 03_Evaluate_Demo_T4.ipynb
├── t4decode/                   # Core Python package
│   ├── cli.py                  # CLI driver and evaluation runner
│   ├── core.py                 # Case generation, scoring, and configuration selector
│   ├── engine.py               # Speculative decoding runtime and KV-cache management
│   └── training.py             # Teacher dataset collection and LoRA trainer
├── tests/                      # Automated unit tests
│   ├── test_core.py
│   ├── test_engine_cpu.py
│   └── test_lora_cpu.py
├── build_notebooks.py          # Notebook packaging and embedding script
├── BUILD_MANIFEST.json         # Cryptographic build manifest
├── requirements-colab.txt      # PyTorch and Hugging Face dependencies
├── pytest.ini                  # Pytest configuration
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## 6. Author and License

- Author: Nhut Nguyen (@nhutnhd)
- License: MIT License (see [LICENSE](LICENSE))
