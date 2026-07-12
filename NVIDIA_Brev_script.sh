#!/bin/bash
# Configure a Brev instance for Isaac GR00T N1.7 finetuning (SO-101 bimanual).
# Target GPU: NVIDIA RTX PRO 6000 (Blackwell, sm_120, 96GB).
# Training / upload / eval commands and rationale: see CHEATSHEET.md.
set -euo pipefail

REPO_DIR="${GROOT_REPO_DIR:-$HOME/Isaac-GR00T}"
GROOT_REPO_URL="${GROOT_REPO_URL:-https://github.com/maxshen1212/Isaac-GR00T.git}"
GROOT_BRANCH="${GROOT_BRANCH:-n1.7-graphen}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"

# Public HF dataset (LeRobot v3.0 — converted to v2.1 below).
DATASET_REPO_ID="${DATASET_REPO_ID:-ChihHanShen/bimanual-so101-pickvials}"
DATASET_ROOT="$REPO_DIR/datasets"
DATASET_PATH="$DATASET_ROOT/$(basename "$DATASET_REPO_ID")"

# Checkpoints live inside the repo so they survive reboots (not /tmp).
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/so101_bimanual_finetune}"

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
        log "ERROR: GPU still unavailable (likely CUDA error 803). Blackwell/CUDA 12.8 needs driver >= 570."
        log "  Fix: reboot the instance, or use a base image with a newer driver."
        nvidia-smi || true
        exit 1
    fi
fi

# ── Phase 4: Dataset download + prep ─────────────────────────────────────────
log "Phase 4: Dataset download + prep"

# 4a. Download (public dataset, no token needed).
if [ ! -f "$DATASET_PATH/meta/info.json" ]; then
    log "Downloading $DATASET_REPO_ID -> $DATASET_PATH"
    uv run hf download "$DATASET_REPO_ID" --repo-type dataset --local-dir "$DATASET_PATH"
else
    log "Dataset already present at $DATASET_PATH"
fi

# 4b. GR00T reads LeRobot v2.1 only; convert v3.0 in place (backs up to <dataset>_v30).
if grep -q '"v3.0"' "$DATASET_PATH/meta/info.json"; then
    log "Converting LeRobot v3.0 -> v2.1"
    GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion --python 3.10 \
        python scripts/lerobot_conversion/convert_v3_to_v2.py \
        --repo-id "$(basename "$DATASET_REPO_ID")" \
        --root "$DATASET_ROOT"
else
    log "Dataset already v2.1 — skipping conversion"
fi

# 4c. Install bimanual modality.json (not shipped with the dataset).
log "Installing bimanual modality.json"
cp examples/SO101_bimanual/modality.json "$DATASET_PATH/meta/modality.json"

# 4d. Generate meta/relative_stats.json (required by N1.7 for custom embodiments).
log "Generating dataset stats"
uv run python gr00t/data/stats.py \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# ── Phase 5: Verification ────────────────────────────────────────────────────
log "Phase 5: Verification"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from gr00t.data.embodiment_tags import EmbodimentTag; print('GR00T imports OK')"
python -c "import json; v=json.load(open('$DATASET_PATH/meta/info.json'))['codebase_version']; assert v=='v2.1', f'dataset is {v}'; print(f'Dataset version OK: {v}')"
python -c "import os; p='$DATASET_PATH/meta/relative_stats.json'; print('relative_stats.json OK' if os.path.exists(p) else 'WARN: relative_stats.json missing')"

log ""
log "============================================================"
log " Setup complete. Next steps: see CHEATSHEET.md (STEP 1 onward)."
log "============================================================"
log ""
log "First (once): hf auth login  +  wandb login   # see CHEATSHEET.md STEP 1"
log "Finetune + auto-upload on success:"
log ""
echo "  cd $REPO_DIR && source .venv/bin/activate && export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \\"
echo "  MAX_STEPS=30000 SAVE_STEPS=5000 SAVE_TOTAL_LIMIT=6 GLOBAL_BATCH_SIZE=64 \\"
echo "  uv run bash examples/finetune.sh \\"
echo "    --base-model-path $BASE_MODEL \\"
echo "    --dataset-path $DATASET_PATH \\"
echo "    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \\"
echo "    --embodiment-tag NEW_EMBODIMENT \\"
echo "    --output-dir $OUTPUT_DIR \\"
echo "    --wandb-project so101-bimanual \\"
echo "    --experiment-name pickvials-n1p7-run1 \\"
echo "  && uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \\"
echo "       $OUTPUT_DIR . --repo-type model --exclude '*optimizer.pt'"
