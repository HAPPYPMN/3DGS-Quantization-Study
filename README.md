# 📦 3DGS-Quantization-Study

A research-oriented implementation and evaluation toolkit for quantization and compression strategies in **3D Gaussian Splatting (3DGS)** pipelines.

This repository supports analyzing the impact of quantization on SH color representation, scale and rotation, and implements per-Gaussian importance evaluation strategies for lossy compression.

## 🚀 Getting Started

To set up the environment, follow these steps:
```bash
conda create --name 3dgs_arch python=3.10
conda activate 3dgs_arch
```
Make sure you install all necessary dependencies before running the simulator.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
