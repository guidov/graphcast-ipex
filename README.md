# GraphCast-Lite: High-Performance GPU-Accelerated Weather Forecasting Inference Engine

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![TensorRT](https://img.shields.io/badge/TensorRT-Optimized-orangered.svg)](https://developer.nvidia.com/tensorrt)

> **⚡ Run a 10-day global 0.25° high-resolution weather forecast in just 40 seconds!**
> 
> **⚙️ VRAM Miracle: Run the top-tier 0.25° global high-resolution physical model using only 8G of VRAM on consumer GPUs (such as RTX 2080 Ti, RTX 3070, RTX 4060, etc.) and extract Google's official TPU v4 level peak inference throughput!**
> 
> **GraphCast-Lite** was born out of an extreme "anti-over-encapsulation" philosophy. We have shattered the high compute and VRAM barriers of weather models. Through smart **Turbo** GNN autoregressive operator restructuring, dual-end spatial chunking, and hard-coded static VRAM locking, we say goodbye to the era when running global weather models required expensive industrial GPUs (like A100/H100), enabling consumer-grade devices to deliver devastatingly fast performance!

---

Traditional weather model frameworks are often accompanied by suffocating over-encapsulation. From the official black-box JAX/Haiku implementation to various bloated GNN libraries (like PyG or DGL), they suffer from poor readability, high performance overhead, and VRAM fragmentation.

This project isolates the complex spherical multi-scale mesh data preparation to basic scientific libraries (`NumPy` and `SciPy`), and builds the core GNN using pure, clean `PyTorch`. Building on this, the **Turbo** sub-module manually decouples the computation graph, serializes custom TensorRT engines, and manages a static memory pool, squeezing hardware overhead to the absolute limit.

---

## ✨ Core Engineering & Key Features

### 🍃 GraphCast-Lite Core: Streamlined Design & Multi-Backend Alignment
*   **Zero Bloat Dependencies**: Grid partition, geographic mappings, and sparse graph building are implemented purely with `NumPy` and `SciPy.spatial.cKDTree`. No need to install massive spatial databases or graph learning frameworks (like PyG).
*   **Pure PyTorch Core**: The model backbone uses standard PyTorch, throwing away complex graph objects and black-box operators. The core logic is clear and easily extendable for fine-tuning and debugging.
*   **Ultra-Lightweight Data Pipeline (`ncutil`)**: Replaces heavy weather data tools (like `xarray` and `dask`). We read variables directly via native `netCDF4` slice operations, simplifying data normalization and reducing I/O latency to near zero.
*   **High-Precision Solar Integration (`forcefield`)**: Instead of the discrete time-step approximations used in DeepMind's original implementation, we use a **fully analytical integration scheme** for Top-Of-Atmosphere (TOA) solar radiation forcing. This guarantees physical conservation, improves accuracy, and boosts calculations by **over 100x**, eliminating CPU bottlenecks.
*   **Multi-Backend Numerical Consistency**: Provides identical, numerically aligned baselines (pure PyTorch, NumPy/CuPy, ONNX Runtime) to ensure outputs match exactly during autoregressive rollouts.

### 🚀 Turbo Sub-Module: Hardware Synergy & Peak VRAM Optimization
The `turbo/` directory houses the high-performance production engine designed to extract maximum throughput:
*   **Static Virtual VRAM Pool (Zero-Allocation VRAM Pool)**: Uses a custom `PoolManager`. It allocates a single, contiguous physical VRAM buffer during initialization. It reuse memory in-place during the autoregressive loop using pre-calculated offsets. **VRAM allocation calls during rollout are exactly zero**, completely avoiding VRAM fragmentation OOMs.
*   **Multi-Subgraph TensorRT Decoupling**: Decouples the massive GraphCast model into 5 highly convergent subgraphs: `MeshEncoder`, `G2MAggregate`, `M2MProcessor`, `M2GInteraction`, and `ForceField`. Subgraph kernel fusion eliminates GPU kernel launch latencies.
*   **Dual-End Spatial Chunking**:
    *   *Grid-to-Mesh Edge Chunking (`echunk`)*: Chunks the sparse G2M edge message passing step using a sliding window.
    *   *Mesh-to-Grid Grid Chunking (`gchunk`)*: Maps the GNN nodes back to the 2D geographical lat-lon grid in batches.
    *   **VRAM Breakthrough**: Using this streaming chunked design, the 0.25° top-tier model runs under **8G of VRAM** in native Float32 precision!
*   **Direct Pointer Binding & Zero-Copy**: Reuses GPU pointers (`.data.ptr`) between CuPy arrays and TensorRT context. State updates during the autoregressive loop are done by swapping physical pointers, achieving **0-byte device memory copies**.

---

## 📥 Model Weights & Test Data Resources

> ⚠️ **Important: The standard Lite mode and the Turbo acceleration mode use different weights.**
> *   **Lite Mode** uses standard PyTorch weights (`.pth` / `.pkl`), suitable for quick validation and fine-tuning.
> *   **Turbo Mode** uses compiled ONNX subgraphs and runtime configuration definitions.
> 
> Download the files and extract them into the `weights/` directory matching the structural paths below:

| Category | Description | Download Link |
| :--- | :--- | :--- |
| **0.25° Test Dataset** | Raw NetCDF input. Due to size, download directly from Google Cloud | [📥 Download ERA5 0.25° Test Data](https://storage.cloud.google.com/dm_graphcast/graphcast/dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc) |
| **General Test Data** | 1.0°/0.25° basic inputs, means, and std vectors (Required) | [📥 Download testdata_and_normvector.zip](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/testdata_and_normvector.zip) |
| **Lite Weights** (PyTorch) | 1.0° model for PyTorch/NumPy/CuPy verification | [📥 Download lite_5mesh_13level_1deg.zip](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/lite_5mesh_13level_1deg.zip) |
| **Lite Weights** (PyTorch) | 0.25° high-res model for PyTorch verification | [📥 Download lite_6mesh_37level_0.25deg.zip](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/lite_6mesh_37level_0.25deg.zip) |
| **Turbo Weights** (TensorRT) | 1.0° serialized subgraphs and config files | [📥 Download turbo_5mesh_13level_1deg.zip](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/turbo_5mesh_13level_1deg.zip) |
| **Turbo Weights** (TensorRT) | 0.25° serialized subgraphs and config files | [📥 Download turbo_6mesh_37level_0.25deg.zip](https://github.com/VectorElectron/graphcast-lite/releases/download/weights/turbo_6mesh_37level_0.25deg.zip) |

> 📌 **Path Alignment**: Keep test datasets in `weights/testdata_and_normvector/` and model directories aligned as `weights/lite_5mesh_13level_1deg/...` or `weights/turbo_6mesh_37level_0.25deg/...`.

---

## 🛠️ Environment Configurations (Layered)

Configure the dependencies depending on your specific hardware and development goal:

### Layer 1: Core Algorithm Validation (NumPy / CuPy)
Suitable for quick validation without heavy frameworks. 
```bash
# Core NetCDF parser and geometric calculations
pip install netCDF4 scipy

# Pure CPU relies on NumPy (stdlib). Install GPU backends matching your local cuda (e.g. CUDA 12.x)
pip install cupy-cuda12x
```

### Layer 2: PyTorch Verification & Fine-Tuning
Enables PyTorch training, model exportation, and backend checking:
```bash
# Install PyTorch and ONNX Runtime
pip install torch onnxruntime
```

### Layer 3: Turbo Production Inference (TensorRT)
Required for multi-subgraph static memory pool and streaming chunking:
```bash
# Install TensorRT (v10 is recommended; v11 requires different build parameters)
pip install tensorrt==10.6.0
```

---

## 🍃 Getting Started: GraphCast-Lite (Baseline PyTorch)

The Lite mode is designed for clean, readable PyTorch code, ideal for developer verification.

#### 1.1 Run PyTorch Inference
```bash
cd infer/
python infer_torch.py
```

#### 1.2 Run pure NumPy / CuPy Verification
Enables seamless execution on either CPU (NumPy) or GPU (CuPy) to verify identical values:
```bash
python infer_numpy.py
```

#### 1.3 Start Minimal Training
```bash
cd graphcast/
python train.py
```

---

## 🚀 Getting Started: Turbo Engine (TensorRT)

The Turbo engine targets production environments, compile custom model layers into high-speed TensorRT engines.

#### 2.1 Compile Subgraphs and VRAM Profiling
Convert ONNX sub-models into TensorRT engine files. The compiler profiles memory usage and automatically writes the VRAM offsets back into `GraphCastInfer.json`:
```bash
# Build subgraphs with FP16 precision
python turbo/model_compile.py --onnx-dir ../weights/para_5mesh_13level_1deg --fp16
```
*(Note: The solar radiation `forcefield` always stays FP32 during compilation to prevent values from overflowing due to step changes.)*

#### 2.2 Run Static Memory Pool Autoregressive Inference
Run the production pipeline using zero-copy and static memory allocation:
```bash
python turbo/infer_tensorrt.py
```

---

## 📊 Turbo Benchmarks on Various Hardware

Autoregressive predictions for a 10-day global weather forecast (40 steps rollout):

| GPU | Mode / Precision | Inference Time (10-Day Forecast) | Peak VRAM Usage | Memory Profile |
| :--- | :---: | :---: | :---: | :---: |
| **RTX 2080 Ti** | Turbo TRT (FP32) | 115 s | ~ 7.8 GB | Constant / Flat |
| **RTX 2080 Ti** | Turbo TRT (FP16) | 40 s | ~ 6.0 GB | Constant / Flat |
| **RTX 4090** | Turbo TRT (TF32) | 29 s | ~ 7.8 GB | Constant / Flat |
| **RTX 4090** | Turbo TRT (BF16) | 15 s | ~ 6.0 GB | Constant / Flat |

> 💡 **Benchmark Details:**
> *   **FP32 Global Model on 8G VRAM**: Standard GraphCast implementations require 40GB+ VRAM for a 0.25° BF16 model. Under the Turbo engine, dual-end spatial chunking and in-place static memory allocation enable FP32 full-precision rollout in under 8GB VRAM.
> *   **Hard-Coded Memory Lock**: The memory usage presents as a flat straight line throughout the forecast duration. There is zero memory churn, eliminating OOM risk.
> *   **High-Throughput on 4090**: Running in BF16, each forecast step takes only **0.375 seconds**, completing a full 10-day global prediction in 15 seconds.

---

## 🔮 Future Outlook: 0.1° Convection-Permitting Weather Prediction

Stepping from 1.0° to 0.25° proved the capability of GraphCast-Lite. Our next frontier is **global 0.1° high-resolution weather forecasting**:
*   **Theoretical Memory Profile**: 0.1° yields an exponential increase in node counts. By upgrading our pipeline to **Multi-Level Cascade Chunking** and **Dynamic Tensor Rematerialization (DTR)**, we aim to lock peak VRAM usage under **24GB**.
*   **Consumer GPU Democratization**: This means researchers can run 0.1° global predictions locally using consumer GPUs (like RTX 3090 / 4090) instead of expensive enterprise multi-GPU clusters.

---

## 🤝 Collaboration & Non-Standard Customization

The ONNX subgraph compilation and computing graph restructuring in the Turbo engine require complex manual tensor arrangements. We welcome collaboration on:
*   **Custom Fine-tuned Weights**: If you have custom fine-tuned regional or specific-variable GraphCast weights, we can help integrate them into the Turbo engine for high-speed local inference.
*   **Alternative Architectures (e.g. GenCast)**: We can adapt our chunking, zero-copy, and static VRAM optimization tools for other large GNN-based physical models experiencing memory limitations.
*   **Non-NVIDIA Hardware Deployment**: Porting the static VRAM engine design to AMD GPUs, Intel XPU/GPUs, or edge devices.

> 📩 **Contact**: Open an issue or contact us through the repository home details.

---

## 🤝 Contributing & License

We want to provide the community with a clean, lightweight, and performant GNN weather pipeline. Contributions on physical operators and memory optimization are welcome!

This project is licensed under the **BSD 3-Clause License**.
