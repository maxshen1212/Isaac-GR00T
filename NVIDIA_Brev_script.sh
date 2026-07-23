#!/bin/bash
# Environment setup only: deps, repo, venv, GPU. No credentials, no data, no training.
# Paste into Brev's Setup Script field. Everything after this: CHEATSHEET.md.
# Target: 8x H100 SXM5 (Hopper, sm_90, 74GB) / 112 CPU / 894 GiB RAM.
#
# Deliberately does NOT fetch the dataset: that step runs after `hf auth login`
# (CHEATSHEET.md STEP 1), which keeps HF credentials out of this file and, because
# the download is then authenticated, sidesteps the per-IP 429 that anonymous
# pulls hit on shared cloud IPs.
set -euo pipefail

REPO_DIR="${GROOT_REPO_DIR:-$HOME/Isaac-GR00T}"
GROOT_REPO_URL="${GROOT_REPO_URL:-https://github.com/maxshen1212/Isaac-GR00T.git}"
GROOT_BRANCH="${GROOT_BRANCH:-n1.7-graphen}"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

log() { echo "==> $*"; }

# ── Phase 1: System dependencies ─────────────────────────────────────────────
log "Phase 1: System dependencies"

$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    ffmpeg libaio-dev git-lfs build-essential curl

git lfs install

# torchcodec (N1.7's video backend) supports FFmpeg 4–7 only, not 8.
FFMPEG_MAJOR="$(ffmpeg -version 2>/dev/null | head -1 | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' || echo '')"
if [ -n "$FFMPEG_MAJOR" ] && [ "$FFMPEG_MAJOR" -ge 8 ] 2>/dev/null; then
    log "WARN: FFmpeg $FFMPEG_MAJOR detected; torchcodec needs 4–7, video decode may fail."
fi

# CUDA toolkit 12.8 (nvcc for deepspeed JIT).
if ! dpkg -s cuda-toolkit-12-8 &>/dev/null; then
    UBUNTU_VERSION=$(. /etc/os-release && echo "${VERSION_ID//.}")
    if ! apt-cache show cuda-toolkit-12-8 &>/dev/null; then
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb"
        log "Adding NVIDIA CUDA apt repository..."
        curl -fsSL "$KEYRING_URL" -o /tmp/cuda-keyring.deb
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        $SUDO apt-get update -qq
    fi
    $SUDO apt-get install -y --no-install-recommends cuda-toolkit-12-8
else
    log "cuda-toolkit-12-8 already installed"
fi

# forward-compat libs — only applied in Phase 3.5 if the driver can't see the GPU.
if ! dpkg -s cuda-compat-12-8 &>/dev/null; then
    $SUDO apt-get install -y --no-install-recommends cuda-compat-12-8
else
    log "cuda-compat-12-8 already installed"
fi
COMPAT_LINE='export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'

# CUDA_HOME for deepspeed.
grep -qF "CUDA_HOME" "$HOME/.bashrc" 2>/dev/null || echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> "$HOME/.bashrc"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

# uv
if ! command -v uv &>/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
grep -qF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
if ! command -v uv &>/dev/null; then
    log "ERROR: uv not found in PATH or ~/.local/bin."
    exit 1
fi

# ── Phase 2: Clone Isaac-GR00T ───────────────────────────────────────────────
log "Phase 2: Clone Isaac-GR00T ($GROOT_REPO_URL @ $GROOT_BRANCH)"

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --recurse-submodules "$GROOT_REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout "$GROOT_BRANCH"
else
    log "Repository already cloned at $REPO_DIR"
    cd "$REPO_DIR"
    git fetch origin "$GROOT_BRANCH"
    git checkout "$GROOT_BRANCH"
    git pull --ff-only origin "$GROOT_BRANCH" || log "Could not fast-forward; keeping current state"
fi
git submodule update --init --recursive
cd "$REPO_DIR"

# ── Phase 3: Python environment ──────────────────────────────────────────────
log "Phase 3: Python environment"

if [ ! -d "$REPO_DIR/.venv" ]; then
    bash scripts/deployment/dgpu/install_deps.sh   # ffmpeg/CUDA + uv sync + editable install
else
    log ".venv already exists, skipping install_deps.sh"
fi
source "$REPO_DIR/.venv/bin/activate"

# ── Phase 3.5: GPU enablement ────────────────────────────────────────────────
# Apply CUDA forward-compat only if the GPU isn't already visible: forcing it
# when the driver natively supports CUDA 12.8 causes a user/kernel driver
# mismatch (CUDA error 803).
log "Phase 3.5: GPU enablement"

gpu_ok() { python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

if gpu_ok; then
    log "GPU visible with the instance driver — not applying cuda-compat."
else
    log "GPU not visible — trying CUDA forward-compat..."
    export LD_LIBRARY_PATH="/usr/local/cuda-12.8/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if gpu_ok; then
        log "GPU visible via cuda-compat — persisting to ~/.bashrc."
        grep -qF "cuda-12.8/compat" "$HOME/.bashrc" 2>/dev/null || echo "$COMPAT_LINE" >> "$HOME/.bashrc"
    else
        log "ERROR: GPU still unavailable (likely CUDA error 803): driver is too old for CUDA 12.8."
        log "  Fix: reboot the instance, or use a base image with a newer driver."
        nvidia-smi || true
        exit 1
    fi
fi

# ── Phase 4: Verification ───────────────────────────────────────────────────
log "Phase 4: Verification"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from gr00t.data.embodiment_tags import EmbodimentTag; print('GR00T imports OK')"

# Hardware vs. the recipe in CHEATSHEET.md (8 GPUs / ~112 CPU / ~894 GiB RAM).
# num_gpus drives per-device batch (global_batch_size // num_gpus) and DeepSpeed
# gating, so a GPU-count mismatch silently changes the effective batch.
python - <<'PY'
import os, torch
n_gpu = torch.cuda.device_count()
n_cpu = os.cpu_count() or 0
ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
print(f"Hardware: {n_gpu} GPU / {n_cpu} CPU / {ram_gb:.0f} GB RAM")
if n_gpu != 8:
    print(f"  WARN: recipe assumes NUM_GPUS=8, found {n_gpu}."
          f" GLOBAL_BATCH_SIZE is a TOTAL (640 -> 80/GPU); recompute it for {n_gpu} GPUs.")
if n_cpu < 84:
    print(f"  WARN: only {n_cpu} CPUs ({n_cpu/max(n_gpu,1):.1f}/GPU) for 8-GPU dataloading.")
if ram_gb < 700:
    print(f"  WARN: only {ram_gb:.0f} GB RAM.")
PY


log "Setup complete. Next: CHEATSHEET.md STEP 1."
