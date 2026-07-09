#!/bin/bash
# setup_groot_so101.sh — 設定 Brev 執行個體以進行
# Isaac GR00T N1.7 微調（SO-101 雙臂，使用自有的 Hugging Face 資料集）
#
# 目標 GPU：NVIDIA RTX PRO 6000（Blackwell 架構，sm_120，96GB）
# 註：N1.7 已從 N1.6 的 Eagle backbone 換成 Cosmos-Reason2-2B（Qwen3-VL），
#     因此必須使用 N1.7 的程式碼（n1.7-graphen 分支），不能沿用 N1.6。
set -euo pipefail

REPO_DIR="${GROOT_REPO_DIR:-$HOME/Isaac-GR00T}"

# 複製你自己的 fork / 分支 —— 內含 examples/SO101_bimanual/（config + modality.json）。
# NVIDIA 官方 repo 沒有這些雙臂設定檔。
GROOT_REPO_URL="${GROOT_REPO_URL:-https://github.com/maxshen1212/Isaac-GR00T.git}"
GROOT_BRANCH="${GROOT_BRANCH:-n1.7-graphen}"

# N1.7 基礎模型（已正式釋出；若 HF 頁面有授權門檻，需先接受條款並 hf auth login 才能下載）。
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"

# 你自己的公開 Hugging Face 資料集（LeRobot v3.0 —— 下方會轉成 v2.1）。
DATASET_REPO_ID="${DATASET_REPO_ID:-ChihHanShen/bimanual-so101-pickvials}"
DATASET_ROOT="$REPO_DIR/datasets"
DATASET_PATH="$DATASET_ROOT/$(basename "$DATASET_REPO_ID")"

# checkpoint 輸出目錄（放在 repo 內，重開機後仍保留；不要用 /tmp，關機會被清空）。
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/so101_bimanual_finetune}"

# 非 root 時才加 sudo
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

log() { echo "==> $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1：系統相依套件（System dependencies）
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 1: System dependencies"

$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    ffmpeg libaio-dev git-lfs build-essential curl

git lfs install

# torchcodec（N1.7 唯一的影片解碼後端）只支援 FFmpeg 4–7，「不支援 FFmpeg 8」。
# 若這裡裝到 8，訓練時讀影片會失敗 —— 先警告，讓你有機會換版本。
FFMPEG_MAJOR="$(ffmpeg -version 2>/dev/null | head -1 | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' || echo '')"
if [ -n "$FFMPEG_MAJOR" ] && [ "$FFMPEG_MAJOR" -ge 8 ] 2>/dev/null; then
    log "警告：偵測到 FFmpeg $FFMPEG_MAJOR，torchcodec 只支援 4–7，影片解碼可能失敗。"
    log "      解法：改用提供 FFmpeg 4–7 的基底映像，或降級 ffmpeg 套件。"
fi

# CUDA toolkit 12.8（提供 nvcc 給 deepspeed JIT）。
# Blackwell（sm_120）需要 CUDA 12.8 以上 + 驅動 570 以上；RTX PRO 6000 通常搭配 580。
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

# CUDA forward-compat 函式庫 —— 讓舊驅動也能跑 CUDA 12.8 runtime。
if ! dpkg -s cuda-compat-12-8 &>/dev/null; then
    $SUDO apt-get install -y --no-install-recommends cuda-compat-12-8
else
    log "cuda-compat-12-8 already installed"
fi

# 註：cuda-compat 只在 Phase 3.5「torch 看不到 GPU 時」才會被啟用。
# 若驅動本來就支援 CUDA 12.8，強行套用 compat 的 libcuda 會造成
# 使用者態/核心態驅動不一致（CUDA error 803）。
COMPAT_LINE='export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'

# 設定 CUDA_HOME 給 deepspeed 使用
CUDA_HOME_LINE='export CUDA_HOME=/usr/local/cuda-12.8'
if ! grep -qF "CUDA_HOME" "$HOME/.bashrc" 2>/dev/null; then
    echo "$CUDA_HOME_LINE" >> "$HOME/.bashrc"
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

# 確保這個 process 裡有可用的 uv
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
# Phase 2：複製 Isaac-GR00T（你的 fork / N1.7 雙臂分支）
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
# Phase 3：Python 環境（用 repo 自帶的安裝腳本）
# ─────────────────────────────────────────────────────────────────────────────
# N1.7 的 dgpu/install_deps.sh 會做：安裝 ffmpeg/CUDA toolkit、uv sync
# （torch 2.7.1+cu128）、以及 uv pip install -e .
log "Phase 3: Python environment"

if [ ! -d "$REPO_DIR/.venv" ]; then
    bash scripts/deployment/dgpu/install_deps.sh
else
    log ".venv already exists, skipping install_deps.sh"
fi

# 啟用 venv 供後續 phase 使用
source "$REPO_DIR/.venv/bin/activate"

# 註：若之後 flash-attn 在 Blackwell（sm_120）報 "no kernel image available for device"，
#     需針對 cu128 重新編譯 flash-attn（預編 wheel 可能還沒支援 Blackwell）。

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.5：GPU 啟用（只有真的需要時才套用 CUDA forward-compat）
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
        # Blackwell（RTX PRO 6000）需要驅動 570 以上（建議 580）。
        log "ERROR: GPU still unavailable (likely CUDA error 803: driver mismatch)."
        log "  Blackwell/CUDA 12.8 needs NVIDIA driver >= 570. Diagnose with 'nvidia-smi'."
        log "  Fixes: reboot the instance (reloads the kernel module), or pick a base image with a newer driver."
        nvidia-smi || true
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4：資料集下載 + 前處理（SO-101 雙臂）
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 4: Dataset download + prep"

# 4a. 從 Hugging Face 下載資料集（公開 → 不需 token）。
if [ ! -f "$DATASET_PATH/meta/info.json" ]; then
    log "Downloading $DATASET_REPO_ID -> $DATASET_PATH"
    uv run hf download "$DATASET_REPO_ID" \
        --repo-type dataset --local-dir "$DATASET_PATH"
else
    log "Dataset already present at $DATASET_PATH"
fi

# 4b. GR00T 的 loader 只讀 LeRobot v2.1；若是 v3.0 就原地轉換。
#     原始版本會自動備份為 <dataset>_v30。
if grep -q '"v3.0"' "$DATASET_PATH/meta/info.json"; then
    log "Dataset is LeRobot v3.0 — converting to v2.1"
    GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion \
        python scripts/lerobot_conversion/convert_v3_to_v2.py \
        --repo-id "$(basename "$DATASET_REPO_ID")" \
        --root "$DATASET_ROOT"
else
    log "Dataset already v2.1 (or non-v3.0) — skipping conversion"
fi

# 4c. 安裝雙臂 modality.json（必要；資料集本身不含）。
log "Installing bimanual modality.json"
cp examples/SO101_bimanual/modality.json "$DATASET_PATH/meta/modality.json"

# 4d.（N1.7 新增步驟）為自訂 embodiment 產生正規化統計。
#     N1.7 訓練會用到 meta/relative_stats.json；先透過 modality config 註冊
#     NEW_EMBODIMENT，再計算統計，避免訓練時因統計缺失/長度不符而報錯。
log "Generating dataset stats (N1.7 requirement for custom embodiment)"
uv run python gr00t/data/stats.py \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5：驗證（Verification）
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 5: Verification"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from gr00t.data.embodiment_tags import EmbodimentTag; print('GR00T imports OK')"
python -c "import json; v=json.load(open('$DATASET_PATH/meta/info.json'))['codebase_version']; assert v=='v2.1', f'dataset is {v}, expected v2.1'; print(f'Dataset codebase_version OK: {v}')"
# 確認 N1.7 統計檔已產生
python -c "import os; p='$DATASET_PATH/meta/relative_stats.json'; print('relative_stats.json OK' if os.path.exists(p) else 'WARN: relative_stats.json missing (stats.py 可能失敗)')"

log ""
log "============================================================"
log " Setup complete. 環境已就緒，可進行 N1.7 SO-101 雙臂微調。"
log "============================================================"
log ""
log "微調前置作業（首次執行會下載 N1.7 模型；如有授權門檻需先接受條款）："
log ""
log "  # (once) 先接受 https://huggingface.co/$BASE_MODEL 的授權條款，"
log "  #        再用 WRITE token 登入，才能下載模型："
log "  #   hf auth login          # ...或 export HF_TOKEN=<token>"
log "  # (once) 讓訓練日誌上傳到 wandb.ai："
log "  #   wandb login <YOUR_API_KEY>     # key: https://wandb.ai/authorize"
log "  # 想關閉 wandb：在指令前加 USE_WANDB=0，並移除 --wandb-* 參數。"
log ""
log "微調指令（N1.7）："
log ""
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \\"
echo "    --base-model-path $BASE_MODEL \\"
echo "    --dataset-path $DATASET_PATH \\"
echo "    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \\"
echo "    --embodiment-tag NEW_EMBODIMENT \\"
echo "    --output-dir $OUTPUT_DIR \\"
echo "    --wandb-project so101-bimanual \\"
echo "    --experiment-name pickvials-n1p7-run1"
log ""
log "上傳訓練好的 checkpoint 到 Hugging Face："
log ""
log "  # checkpoints 放在 repo 內（重開機仍在），但整台執行個體被刪除前仍要先上傳。"
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  # 先看有哪些 checkpoint： ls $OUTPUT_DIR"
echo "  uv run hf upload \\"
echo "    ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \\"
echo "    $OUTPUT_DIR/checkpoint-20000 \\"
echo "    . \\"
echo "    --repo-type model      # 加 --private 可保持不公開"
log ""
log "訓練後做 open-loop 評估："
log ""
echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  uv run python gr00t/eval/open_loop_eval.py \\"
echo "    --dataset-path $DATASET_PATH \\"
echo "    --embodiment-tag NEW_EMBODIMENT \\"
echo "    --model-path $OUTPUT_DIR/checkpoint-20000 \\"
echo "    --traj-ids 0 --action-horizon 16 --steps 400"
