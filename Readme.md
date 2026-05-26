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
pi-gpu/
├── pi_gpu.c       # Complete program (host + kernels)
├── README.md      # This file
└── pi_*_digits.txt # Generated output


## Key Design Decisions

1. **C instead of Fortran** — OpenCL host code is natively C, and mixing Fortran + OpenCL + GMP interop would be fragile. C gives direct access to the OpenCL API without binding layers.

2. **Two-phase approach** — The GPU handles the embarrassingly parallel leaf evaluation (thousands of independent terms), while the CPU handles the merge tree where dependencies make GPU parallelism harder to exploit.

3. **32-bit limbs in GPU kernels** — RDNA 2's native 32-bit integer multiply produces a 64-bit result via `ulong`, avoiding the need for multi-precision arithmetic within the kernel itself.

4. **Unified memory** — The Steam Deck's APU shares LPDDR5 between CPU and GPU. Using `CL_MEM_ALLOC_HOST_PTR` means zero-copy access, eliminating the biggest bottleneck in discrete-GPU setups.

5. **Batch size of 4096** — Balances GPU occupancy (enough work-items to fill 8 CUs) against the per-dispatch overhead. Each batch of 4096 terms produces ~56,000 digits worth of partial results.
