# GraphCast-Lite: Running on Intel Arc GPUs (XPU / IPEX)

This guide details the setup and execution workflow for running **GraphCast-Lite** (a lightweight PyTorch port of Google DeepMind's GraphCast weather model) on an **Intel Arc B580 GPU** (12GB VRAM) under Linux.

---

## 💻 System Configuration & Environment

The environment configuration validated for this setup:
*   **Operating System**: Linux
*   **Target GPU**: Intel Arc B580 (12GB VRAM, Battlemage Architecture)
*   **Conda Environment**: `ipex` (located at `/home/guido/anaconda3/envs/ipex`)
*   **Python Version**: `3.12`
*   **PyTorch Backend**: `2.6.0+xpu` (with native `torch.xpu` support)

---

## 🛠️ Step-by-Step Installation

### 1. Environment & Packages
Activate the `ipex` Conda environment and install the required weather data file libraries (`netCDF4` and `scipy`):

```bash
conda activate ipex
pip install netCDF4 scipy
```

### 2. Download Model Weights & Data
Prepare the directories and download the 1.0° resolution model weights and the baseline test dataset:

```bash
mkdir -p weights
cd weights

# Download Test Data (87MB)
wget https://github.com/VectorElectron/graphcast-lite/releases/download/weights/testdata_and_normvector.zip
unzip testdata_and_normvector.zip

# Download 1.0° Model Weights (389MB)
wget https://github.com/VectorElectron/graphcast-lite/releases/download/weights/lite_5mesh_13level_1deg.zip
unzip lite_5mesh_13level_1deg.zip

cd ..
```

---

## 🚀 How to Run

Run the PyTorch inference script using the `ipex` environment Python:

```bash
conda activate ipex
cd infer
python infer_torch.py
```

---

## 💡 Engineering & Optimization Highlights

To run this model successfully and performantly on Intel GPUs, two critical code changes were introduced:

### 1. The Conda OpenCL/oneDNN Driver Override Bug
When you run `conda activate ipex`, Conda runs a library script `activate-opencl-rt.sh` under the hood. This script overrides `OCL_ICD_VENDORS` to point inside the virtual environment:
`OCL_ICD_VENDORS=/home/guido/anaconda3/envs/ipex/etc/OpenCL/vendors`

Because the virtual environment lacks the physical host driver mappings for the Intel B580 GPU (which reside in `/etc/OpenCL/vendors`), PyTorch's linear algebra engine (oneDNN) fails to load GPU kernels, throwing:
`RuntimeError: could not create an engine`

**The Fix:**
We added an environment cleanup block at the top of `infer/infer_torch.py` that deletes these overrides from `os.environ` before PyTorch initializes, allowing oneDNN to fall back to the host system drivers:
```python
import os
# Fix OpenCL ICD override bug in Conda environment
for var in ['OCL_ICD_VENDORS', 'OCL_ICD_VENDORS_RESET', 'OCL_ICD_FILENAMES_RESET']:
    if var in os.environ:
        del os.environ[var]
```

### 2. Gradient Checkpoint Bypass (21x Speedup)
The default GraphCast model code wraps GNN sub-blocks inside `torch.utils.checkpoint` to save memory during training. 

During inference (`model.eval()`), checkpointing is redundant. On the Intel XPU compiler backend, wrapping linear layers in checkpointing triggers initialization bottlenecks and forces slow emulation execution paths.

**The Fix:**
We updated `graphcast/graphcast.py` to check if the model is training, and execute standard evaluation blocks when `self.training` is False:
```python
        if self.training:
            g2m_gf_out, g2m_mf_out = checkpoint(self.grid2mesh, x, use_reentrant=False)
            m2m_mf_out = checkpoint(self.mesh2mesh, g2m_mf_out, use_reentrant=False)
            dynamic_delta = checkpoint(self.mesh2grid, m2m_mf_out, g2m_gf_out, use_reentrant=False)
        else:
            g2m_gf_out, g2m_mf_out = self.grid2mesh(x)
            m2m_mf_out = self.mesh2mesh(g2m_mf_out)
            dynamic_delta = self.mesh2grid(m2m_mf_out, g2m_gf_out)
```
*   **Before:** ~53.1 seconds per 4-step forecast.
*   **After:** **~2.4 seconds** per 4-step forecast (a **21x speedup**!).
