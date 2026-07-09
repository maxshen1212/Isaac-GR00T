#!/bin/bash
# setup_groot_so101.sh — Configure a Brev instance for
# Isaac GR00T N1.6 post-training (SO-101 BIMANUAL, own HF dataset)
set -euo pipefail

REPO_DIR="${GROOT_REPO_DIR:-$HOME/Isaac-GR00T}"

# Clone YOUR fork/branch — it contains examples/SO101_bimanual/ (config + modality.json)
# and the launch_finetune tweak. NVIDIA upstream does NOT have these.
GROOT_REPO_URL="${GROOT_REPO_URL:-https://github.com/maxshen1212/Isaac-GR00T.git}"
GROOT_BRANCH="${GROOT_BRANCH:-n1.6-graphen}"

# Your own public Hugging Face dataset (LeRobot v3.0 — converted to v2.1 below).
DATASET_REPO_ID="${DATASET_REPO_ID:-ChihHanShen/bimanual-so101-pickvials}"
DATASET_ROOT="$REPO_DIR/datasets"
DATASET_PATH="$DATASET_ROOT/$(basename "$DATASET_REPO_ID")"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

log() { echo "==> $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: System dependencies
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 1: System dependencies"

$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    ffmpeg libaio-dev git-lfs build-essential curl

git lfs install

# CUDA toolkit 12.8 (provides nvcc for deepspeed JIT)
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

# CUDA forward-compat library — allows CUDA 12.8 runtime on driver 550 (H100)
if ! dpkg -s cuda-compat-12-8 &>/dev/null; then
    $SUDO apt-get install -y --no-install-recommends cuda-compat-12-8
else
    log "cuda-compat-12-8 already installed"
fi

# NOTE: CUDA forward-compat (cuda-compat-12-8) is applied CONDITIONALLY in Phase 3.5,
# ONLY if torch cannot see the GPU with the instance's own driver. Forcing the compat
# libcuda onto a driver that already supports CUDA 12.8 causes a user-mode/kernel-mode
# driver mismatch (CUDA error 803) — that is what broke Phase 5 on the first run.
COMPAT_LINE='export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'

# Also ensure CUDA_HOME is set for deepspeed
CUDA_HOME_LINE='export CUDA_HOME=/usr/local/cuda-12.8'
if ! grep -qF "CUDA_HOME" "$HOME/.bashrc" 2>/dev/null; then
    echo "$CUDA_HOME_LINE" >> "$HOME/.bashrc"
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

# Ensure uv exists and is callable in this process.
if ! command -v uv &>/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
UV_PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if ! grep -qF "$UV_PATH_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$UV_PATH_LINE" >> "$HOME/.bashrc"
fi

if command -v uv &>/dev/null; then
    UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
else
    log "ERROR: uv is not available in PATH and ~/.local/bin/uv was not found."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Clone Isaac-GR00T (your fork / bimanual branch)
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 2: Clone Isaac-GR00T ($GROOT_REPO_URL @ $GROOT_BRANCH)"

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --recurse-submodules "$GROOT_REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout "$GROOT_BRANCH"
    git submodule update --init --recursive
else
    log "Repository already cloned at $REPO_DIR"
    cd "$REPO_DIR"
    git fetch origin "$GROOT_BRANCH"
    git checkout "$GROOT_BRANCH"
    git pull --ff-only origin "$GROOT_BRANCH" || log "Could not fast-forward; continuing with current state"
    git submodule update --init --recursive
fi

cd "$REPO_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Python environment (via repo's own install script)
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 3: Python environment"

if [ ! -d "$REPO_DIR/.venv" ]; then
    bash scripts/deployment/dgpu/install_deps.sh
else
    log ".venv already exists, skipping install_deps.sh"
fi

# Activate for subsequent phases
source "$REPO_DIR/.venv/bin/activate"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.5: GPU enablement (apply CUDA forward-compat only if actually needed)
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 3.5: GPU enablement"

gpu_ok() { python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

if gpu_ok; then
    log "GPU visible to torch with the instance's own driver — NOT applying cuda-compat."
else
    log "torch cannot see the GPU with the base driver — trying CUDA forward-compat..."
    export LD_LIBRARY_PATH="/usr/local/cuda-12.8/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if gpu_ok; then
        log "GPU visible via cuda-compat — persisting LD_LIBRARY_PATH to ~/.bashrc."
        if ! grep -qF "cuda-12.8/compat" "$HOME/.bashrc" 2>/dev/null; then
            echo "$COMPAT_LINE" >> "$HOME/.bashrc"
        fi
    else
        log "ERROR: GPU still unavailable (likely CUDA error 803: driver mismatch)."
        log "  CUDA 12.8 needs NVIDIA driver >= 570. Diagnose with 'nvidia-smi'."
        log "  Fixes: reboot the instance (reloads the kernel module), or pick a base image with a newer driver."
        nvidia-smi || true
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Dataset download + prep (SO-101 bimanual)
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 4: Dataset download + prep"

# 4a. Download the dataset from Hugging Face (public → no token needed).
if [ ! -f "$DATASET_PATH/meta/info.json" ]; then
    log "Downloading $DATASET_REPO_ID -> $DATASET_PATH"
    uv run hf download "$DATASET_REPO_ID" \
        --repo-type dataset --local-dir "$DATASET_PATH"
else
    log "Dataset already present at $DATASET_PATH"
fi

# 4b. GR00T's loader only reads LeRobot v2.1. Convert in place if v3.0.
#     Original is backed up automatically as <dataset>_v30.
if grep -q '"v3.0"' "$DATASET_PATH/meta/info.json"; then
    log "Dataset is LeRobot v3.0 — converting to v2.1"
    GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion \
        python scripts/lerobot_conversion/convert_v3_to_v2.py \
        --repo-id "$(basename "$DATASET_REPO_ID")" \
        --root "$DATASET_ROOT"
else
    log "Dataset already v2.1 (or non-v3.0) — skipping conversion"
fi

# 4c. Install the bimanual modality.json (required; not shipped in the dataset).
log "Installing bimanual modality.json"
cp examples/SO101_bimanual/modality.json "$DATASET_PATH/meta/modality.json"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Verification
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 5: Verification"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from gr00t.data.embodiment_tags import EmbodimentTag; print('GR00T imports OK')"
python -c "import json; v=json.load(open('$DATASET_PATH/meta/info.json'))['codebase_version']; assert v=='v2.1', f'dataset is {v}, expected v2.1'; print(f'Dataset codebase_version OK: {v}')"

log ""
log "============================================================"
log " Setup complete. Environment is ready for SO-101 bimanual post-training."
log "============================================================"
log ""
log "To finetune on your dataset:"
log ""
log "  # (once) authenticate wandb so training logs stream to wandb.ai:"
log "  #   wandb login <YOUR_API_KEY>        # key from https://wandb.ai/authorize"
log "  #   ...or export WANDB_API_KEY=<key> before launching."
log "  # To disable wandb instead: prefix the command with USE_WANDB=0 and drop the --wandb-* flags."
log ""
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \\"
echo "    --base-model-path nvidia/GR00T-N1.6-3B \\"
echo "    --dataset-path $DATASET_PATH \\"
echo "    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \\"
echo "    --embodiment-tag NEW_EMBODIMENT \\"
echo "    --output-dir /tmp/so101_bimanual_finetune \\"
echo "    --wandb-project so101-bimanual \\"
echo "    --experiment-name pickvials-run1"
log ""
log "To upload a trained checkpoint to Hugging Face:"
log ""
log "  # NOTE: /tmp is wiped when the instance stops — upload before shutting down."
log "  # (once) authenticate with a WRITE token from https://huggingface.co/settings/tokens :"
log "  #   hf auth login          # ...or export HF_TOKEN=<write_token>"
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  # pick a checkpoint that exists: ls /tmp/so101_bimanual_finetune/"
echo "  uv run hf upload \\"
echo "    ChihHanShen/gr00t-n1.6-so101-bimanual-pickvials \\"
echo "    /tmp/so101_bimanual_finetune/checkpoint-10000 \\"
echo "    . \\"
echo "    --repo-type model      # add --private to keep it unlisted"
log ""
log "To run open-loop evaluation after training:"
log ""
echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  uv run python gr00t/eval/open_loop_eval.py \\"
echo "    --dataset-path $DATASET_PATH \\"
echo "    --embodiment-tag NEW_EMBODIMENT \\"
echo "    --model-path /tmp/so101_bimanual_finetune/checkpoint-10000 \\"
echo "    --traj-ids 0 --action-horizon 16 --steps 400"
