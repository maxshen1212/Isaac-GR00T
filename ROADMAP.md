# GR00T N1.6 Fine-tune Pipeline — ROADMAP

> 目標：用自己的 SO-101 **雙臂**資料跑一次 fine-tune，**先驗證訓練 pipeline 沒問題**，
> 確認無誤後再收更多資料、拉長訓練。
> 參考流程：NVIDIA Sim-to-Real SO-101 教學 = repo 內 `examples/SO100/README.md`。

最後更新：2026-07-02

---

## 0. 環境（✅ 已完成）

| 項目 | 狀態 |
|---|---|
| Python | ✅ 3.10.12（**不可用 3.11**：repo 沒提供 3.11 的 flash-attn 預編 wheel，會退回原始碼編譯而缺 nvcc 失敗） |
| torch / torchvision | ✅ 2.7.1+cu128 / 0.22.1+cu128 |
| flash-attn | ✅ 2.7.4.post1（預編 wheel，未編譯） |
| GPU | ✅ RTX 5090（sm_120 / Blackwell），torch + flash-attn forward 實測可跑 |
| gr00t 套件 | ✅ `uv pip install -e .` 完成，可匯入 |

⚠️ **注意**：目前**沒有** `.python-version` 檔。若日後重跑 `uv sync`（未帶 `--python`），
uv 預設偏好 managed Python，可能又落到 3.11。重裝時請用 `uv sync --python 3.10`，
或在專案根目錄放 `.python-version`（內容 `3.10`）釘死。

---

## 1. 資料集現況

- 路徑：`/home/air-420/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials`
- 格式：**LeRobot v3.0**（`meta/info.json` → `codebase_version: v3.0`）
- 規模：9 episodes / 15170 frames / 1 task / fps 30 / robot_type `so101_follower`
- **state / action = 12 維（雙臂）**
  - 左臂 index 0–5：`left_shoulder_pan / lift / elbow_flex / wrist_flex / wrist_roll / gripper`
  - 右臂 index 6–11：`right_shoulder_pan / lift / elbow_flex / wrist_flex / wrist_roll / gripper`
- **3 個鏡頭**：`observation.images.center` / `observation.images.wrist_left` / `observation.images.wrist_right`
- ❌ 尚無 `meta/modality.json`（GR00T 訓練必需）

---

## 2. 待辦（依序執行）

### ☑ Step 1 — 格式轉換 v3.0 → v2.1（**已完成 2026-07-02**）
GR00T 載入器只讀 v2.1。腳本會就地轉換，原始 v3.0 自動備份為 `..._v30`（可回復）。

> ✅ 已驗證：`codebase_version=v2.1`、9 episodes / 15170 frames、3 鏡頭各 9 支影片、
> state/action 皆 12 維、備份 `bimanual-so101-pickvials_v3.0` 保留。

**系統前置（必裝）**：轉換切影片需要 `ffmpeg`（系統工具，非 pip 套件）：
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg   # 驗證：ffmpeg -version
```

```bash
cd /home/air-420/Isaac-GR00T-n1.6-release
GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id bimanual-so101-pickvials \
  --root /home/air-420/Sim-to-Real-SO-101-Workshop/datasets
```

- **不需要**先手動跑 `scripts/lerobot_conversion/README.md` 的 Setup（`uv venv` + `uv pip install -e .`）。
  `uv run --project` 會自動建立並同步該子環境（`lerobot` 已列在它的 `pyproject.toml` 依賴）。
- `GIT_LFS_SKIP_SMUDGE=1`：首次從 git 安裝 lerobot 時避免拉一堆 LFS 大檔導致卡住/失敗。
- 該子專案 `requires-python = ">=3.10,<3.12"`（只吃 3.10/3.11，**不能 3.12**）；系統 3.10.12 會被自動選用。
- 第一次會 clone + 安裝 lerobot（含 torch 等依賴），可能要數分鐘。

驗收：轉換後 `meta/info.json` 的 `codebase_version` 應變成 `v2.1`，且出現 `bimanual-so101-pickvials_v30` 備份資料夾。

### ☑ Step 2 — 建立**雙臂** `meta/modality.json`（**已完成 2026-07-02**）
來源檔：`examples/SO101_bimanual/modality.json`，已複製到資料集 `meta/modality.json`。
- state / action：`left_arm` 0–5、`left_gripper` 5–6、`right_arm` 6–11、`right_gripper` 11–12（12 維全覆蓋 ✅）
- video：`center` / `wrist_left` / `wrist_right` → `observation.images.*`
- annotation：`human.task_description → task_index`

### ☑ Step 3 — 建立**雙臂** data config（**已完成 2026-07-02**）
檔案：`examples/SO101_bimanual/so101_bimanual_config.py`（已驗證可 import + 註冊 NEW_EMBODIMENT）。
- video keys：`center / wrist_left / wrist_right`
- state / action keys：`left_arm / left_gripper / right_arm / right_gripper`
- `action_configs` 共 4 個（每個 group 一個，順序對齊）：arm→RELATIVE、gripper→ABSOLUTE

### ☐ Step 4 — Smoke test（短步數驗證 pipeline）
先跑 ~60–100 步，只驗證「資料載入 → 前/反向 → 存 checkpoint」全通，**不是完整訓練**。

**wandb 前置（一次性）**：到 https://wandb.ai/authorize 取 key 後登入：
```bash
cd /home/air-420/Isaac-GR00T-n1.6-release
.venv/bin/wandb login <你的_API_KEY>        # 驗證：.venv/bin/wandb login --verify
```

**⚠️ 32GB 顯卡記憶體調整（RTX 5090）**：3B 全參數微調在 32GB 上會 OOM（batch=1 也不夠，靜態就 ~29GB）。
已對 `gr00t/experiment/launch_finetune.py` 做兩處本地改動：
- line 70：`load_bf16 = False → True`（權重 fp32→bf16，省 ~6GB；模型本來預設就是 bf16）
- 新增 `config.training.gradient_checkpointing = True`（省 activation）
再配合 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 減少碎片化。

```bash
cd /home/air-420/Isaac-GR00T-n1.6-release
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 USE_WANDB=1 MAX_STEPS=100 SAVE_STEPS=100 \
GLOBAL_BATCH_SIZE=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /home/air-420/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/so101_bimanual_smoketest \
  --wandb-project so101-bimanual \
  --experiment-name smoketest-100steps \
  -- --gradient_accumulation_steps 8
```
驗收：跑完不報錯、`/tmp/so101_bimanual_smoketest/checkpoint-100` 有產生、wandb 上 loss 曲線有在下降。
（不想用 wandb 時改 `USE_WANDB=0` 並移除最後兩個 `--wandb-*` 參數即可。）

**若仍 OOM**：改 8-bit Adam（launch_finetune.py:79 `optim="adamw_torch"` → `"adamw_bnb_8bit"`，Adam 狀態砍半）
或用 DeepSpeed ZeRO 把優化器狀態 offload 到 CPU RAM。

### ☐ Step 5 —（可選）Open-loop 評估
```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /home/air-420/Sim-to-Real-SO-101-Workshop/datasets/bimanual-so101-pickvials \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/so101_bimanual_smoketest/checkpoint-100 \
  --traj-ids 0 --action-horizon 16 --steps 400
```

### ☐ Step 6 — 正式訓練
Smoke test 通過後，收更多資料，並把 `MAX_STEPS` 拉到 ~10000–20000、開 `USE_WANDB=1`、調 `GLOBAL_BATCH_SIZE`。

---

## 3. 關鍵陷阱備忘
- **Python 必須 3.10 / 3.12**，不可 3.11（flash-attn wheel 缺）。
- **資料必須先轉 v2.1**，v3.0 直接餵會讀不到。
- **modality.json / config 必須是雙臂版**，NVIDIA 教學頁面用的是 SO100 單臂範本，與本資料集不符。
- embodiment tag 用 `NEW_EMBODIMENT`（repo 無 so101 專屬 tag）。
- 首次跑會下載 base model `nvidia/GR00T-N1.6-3B`（需要網路 / HF 存取）。
