# GPU-Accelerated Pi Calculator for Steam Deck

Computes as many digits of π as will fit in available RAM, using the
Steam Deck's AMD RDNA 2 GPU for the heavy multi-precision arithmetic
and the Zen 2 CPU cores for the binary splitting tree orchestration.

## Hardware Target

| Component | Steam Deck APU (Aerith) |
|-----------|------------------------|
| CPU       | Zen 2, 4C/8T @ 2.4–3.5 GHz |
| GPU       | RDNA 2, 8 CUs (512 SPs) @ 1.0–1.6 GHz |
| RAM       | 16 GB LPDDR5 (unified) |

The unified memory architecture means the GPU can access system RAM
directly without PCIe transfers — ideal for big-integer workloads.

## Why OpenCL?

- **Available on SteamOS** out of the box (Mesa Clover/Rusticl)
- **Unified memory** via `CL_MEM_ALLOC_HOST_PTR` avoids copies
- **Portable** — also works on NVIDIA, Intel, and other AMD GPUs
- **Simpler** than Vulkan Compute for this use case

## Dependencies

```bash
# SteamOS / Arch Linux
sudo pacman -S opencl-mesa gcc libgmp openmp

# Ubuntu / Debian
sudo apt-get install ocl-icd-opencl-dev gcc libgmp-dev

# Building the program.
gcc -fopenmp -O2 -o pi_gpu pi_gpu.c -lOpenCL -lgmp -lm

# Use all 8 threads + GPU
OMP_NUM_THREADS=8 ./pi_gpu

# Limit to 4 CPU threads
OMP_NUM_THREADS=4 ./pi_gpu

# How it works.
Phase 1: GPU Leaf Evaluation
─────────────────────────────
  For each batch of 4096 Chudnovsky terms:
    GPU kernel computes P(k), Q(k), T(k) in parallel
    Each work-item handles one term k
    Results written to unified memory buffers

Phase 2: CPU Parallel Merge
─────────────────────────────
  Binary splitting merge tree computed on CPU
  OpenMP parallelizes the merge across all threads
  Stride-doubling reduction combines P/Q/T pairs

Phase 3: Final Assembly (CPU)
─────────────────────────────
  π = 426880 × √10005 × Q / (13591409 × Q + T)
  Uses GMP for the final large-integer arithmetic.

# Check available platforms
clinfo
# If using Mesa/Rusticl on SteamOS:
export RUSTICL_ENABLE=radeonsi

# As root on SteamOS
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
