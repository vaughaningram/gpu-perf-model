# Development and Execution Environment

This project separates local development from GPU execution.

## Canonical workflow

```text
Windows development machine
        ↓ git push
GitHub repository
        ↓ git pull
Tufts Pax login node
        ↓ Slurm allocation
NVIDIA GPU compute node
        ↓
build, correctness checks, benchmarks, and profiling
        ↓
curated CSV results and analysis returned through Git
```

The Windows machine is used for editing, source control, Python modeling,
plotting, and documentation. CUDA executables must be built and run on an
allocated Pax compute node, not on the local AMD GPU and not on a Pax login
node.

## Local environment

Inventory recorded on 2026-08-14:

- Windows development machine
- AMD Radeon RX 9070 XT discrete GPU
- AMD Radeon integrated graphics
- No NVIDIA driver or CUDA-capable GPU
- CUDA toolkit, `nvcc`, CMake, and Nsight Compute not currently installed

The absence of a local CUDA toolchain is acceptable because Pax is the
canonical build and execution environment. Local tooling can be added later if
it provides a clear development benefit, but local results must not be mixed
with the canonical GPU measurements.

## Tufts Pax access

The upgraded Pax cluster is accessible through:

- SSH: `login-prod.pax.tufts.edu`
- Web: <https://ondemand-prod.pax.tufts.edu>

Off-campus access may require the Tufts VPN. SSH access also requires an active
HPC account and may require two-factor authentication.

### Verified canonical environment

Inventory completed on 2026-08-15 using an interactive Slurm allocation:

- Compute node during inventory: `pax143`
- GPU: NVIDIA A100 PCIe 40 GB
- Driver: 575.57.08
- Driver-reported CUDA compatibility: 12.9
- CUDA module: `cuda/12.9.0`
- NVCC: 12.9.41
- Host compiler modules: `gcc/12.4.0` and `g++/12.4.0`
- CMake module: `cmake/3.31.6`
- Nsight Compute CLI: 2025.4.0.0, provided by the CUDA module

The initial CMake configuration detected both GNU C++ 12.4.0 and NVIDIA CUDA
12.9.41. The CPU reference library and tests compiled successfully, and the
first CTest run passed on this environment.

From Windows Terminal or PowerShell:

```powershell
ssh YOUR_UTLN@login-prod.pax.tufts.edu
```

Replace `YOUR_UTLN` with the Tufts username. The resulting shell is a login
node. Do not compile, benchmark, profile, or run project software there.

## First cluster inventory

The first session is an environment-discovery exercise. Do not assume module
names or versions before recording what Pax currently provides.

### 1. On the login node

The following commands inspect the session and available modules without
running the project:

```bash
hostname
git --version
module --version
module spider cuda
module spider gcc
module spider cmake
module spider nsight
```

Clone the repository if it is not already present:

```bash
git clone https://github.com/vaughaningram/gpu-perf-model.git
cd gpu-perf-model
```

### 2. Request one GPU interactively

For the initial bring-up, request any available GPU in the regular GPU
partition:

```bash
srun -p gpu --gres=gpu:1 --mem=8g -t 1:00:00 --pty bash
```

The upgraded-cluster documentation currently directs GPU users to the `gpu`
partition. Partition availability and site policy remain authoritative if this
example later becomes stale.

### 3. On the allocated compute node

First capture the assigned hardware and driver:

```bash
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version --format=csv
```

Do not manually set `CUDA_VISIBLE_DEVICES`; Slurm controls GPU visibility.

Next, load the module versions discovered on the login node. The exact module
names are intentionally placeholders until the first session:

```bash
module load cuda/<DISCOVERED_VERSION>
module load gcc/<DISCOVERED_VERSION>
module load cmake/<DISCOVERED_VERSION>  # only if CMake is not already usable
```

Then inventory the build and profiling tools:

```bash
nvcc --version
gcc --version
g++ --version
cmake --version
ncu --version
```

Also record the active module set:

```bash
module list
```

If `ncu` is unavailable, search the module system for Nsight Compute rather
than installing or guessing a toolchain during the first session.

## Initial versus controlled hardware

M0 may use any modern assigned NVIDIA GPU to establish compilation,
correctness, timing, and result-output infrastructure. Every result must record
the actual GPU model and software environment.

The NVIDIA A100 PCIe 40 GB is the primary experimental target. M1 and later
correlation measurements should request this GPU configuration consistently.
M0 bring-up may still use another GPU when the result is clearly labeled and is
not mixed into the primary performance dataset.

Cross-architecture comparisons are optional later work. They must not introduce
uncontrolled hardware variation into the primary result set.

## Reproducibility metadata

Benchmark outputs will eventually record at least:

- Git commit
- Timestamp
- Hostname
- GPU name and compute capability
- GPU UUID where appropriate for raw experiment records
- Driver and CUDA toolkit versions
- Compiler and build type
- Kernel/configuration identifier
- Matrix dimensions
- Warmup and measured iteration counts

Clock state and other hardware controls will be considered when the measurement
methodology is designed; they are not silently assumed during M0 bring-up.

## Official references

- [Tufts cluster login](https://rtguides.it.tufts.edu/hpc/access/20-cli.html)
- [Tufts upgraded cluster](https://rtguides.it.tufts.edu/hpc/examples/new-cluster.html)
- [Tufts GPU resources](https://rtguides.it.tufts.edu/hpc/compute/gpu.html)
