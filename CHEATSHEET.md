# GR00T N1.7 · SO-101 雙臂微調 Cheatsheet

在 NVIDIA Brev(1× RTX PRO 6000 96GB)上,從零到訓練完成並上傳的完整執行順序。

**路徑基準**

| 項目 | 路徑 / 值 |
|---|---|
| Repo | `~/Isaac-GR00T`(fork `maxshen1212/Isaac-GR00T` @ `n1.7-graphen`) |
| 資料集 | `~/Isaac-GR00T/datasets/bimanual-so101-pickvials` |
| 輸出 checkpoint | `~/Isaac-GR00T/checkpoints/so101_bimanual_finetune` |
| Base model | `nvidia/GR00T-N1.7-3B` |
| 上傳目標 repo | `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials` |

---

## STEP 0 — 環境設定(交給 Brev 開機 Setup Script,不用手動跑)

把整份 `NVIDIA_Brev_script.sh` 貼到 Brev 部署頁的 **Setup Script → Paste Script**。VM 開好後 Brev 會自動執行(clone repo、裝依賴、下載+轉換資料集、算 stats、驗證 GPU),所以**不用手動 curl / bash**。

> ⚠️ **部署前必做 — 給開機腳本 `HF_TOKEN`**:開機腳本在你登入前就跑,匿名下載資料集會被
> HF 以「每 IP」限流(共享雲端 IP 很快撞 **HTTP 429**)。在**貼進 Brev 的腳本最上面**(`set -euo pipefail` 之後)
> 加一行 `export HF_TOKEN=hf_xxxxxxxx`(READ token 即可),或用 Brev 的環境變數/secret 機制帶進去。

SSH 進去後,先驗證開機腳本跑完了、而且東西在你的家目錄:

```bash
# 1) 環境是否建好（Brev 頁面也看得到 setup log；若還在跑就等它完成）
ls ~/Isaac-GR00T/.venv/bin/activate && echo "venv OK"
test -f ~/Isaac-GR00T/datasets/bimanual-so101-pickvials/meta/relative_stats.json && echo "dataset+stats OK"

# 2) uv 是否在 PATH（setup 若以 root 身分跑，uv/repo 可能落在 /root 而非你的家目錄）
which uv || ls ~/.local/bin/uv
```

> ⚠️ **落點陷阱**:若第 2 步 `uv` 找不到,檢查 `sudo ls /root/Isaac-GR00T`——若在那,代表 setup 以 root 執行,
> repo/venv/uv 都在 `/root`。解法:後續步驟改用 `sudo -i` 切成 root 操作。

兩項都 OK → 進 STEP 1。

**若開機腳本在資料集下載時 429 掛掉(沒帶 HF_TOKEN):** 環境(Phase 1–3)通常已裝好,只要登入後手動補跑 Phase 4:

```bash
cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"
hf auth login                                                   # 或 export HF_TOKEN=hf_xxx
rm -rf ~/Isaac-GR00T/datasets/bimanual-so101-pickvials         # 清掉半套下載
uv run hf download ChihHanShen/bimanual-so101-pickvials \
    --repo-type dataset --local-dir ~/Isaac-GR00T/datasets/bimanual-so101-pickvials
GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion --python 3.10 \
    python scripts/lerobot_conversion/convert_v3_to_v2.py \
    --repo-id bimanual-so101-pickvials --root ~/Isaac-GR00T/datasets
cp examples/SO101_bimanual/modality.json ~/Isaac-GR00T/datasets/bimanual-so101-pickvials/meta/modality.json
uv run python gr00t/data/stats.py \
    --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py
```

---

## STEP 1 — 登入 HF + wandb(訓練前一次)

```bash
cd ~/Isaac-GR00T
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

# HF（一次性）：接受/申請「兩個」gated repo 的存取，再用 WRITE token 登入
#   1) https://huggingface.co/nvidia/GR00T-N1.7-3B      → 接受授權（base model）
#   2) https://huggingface.co/nvidia/Cosmos-Reason2-2B  → 申請存取（N1.7 的 VLM backbone）
#      ★ 每次載入 GR00T 都會用到這個 backbone，沒授權會在載入模型時 403 GatedRepoError
#        （資料集/base model 下載成功也沒用，會卡在這一步）。核准通常接受條款後即時。
hf auth login                 # 或 export HF_TOKEN=<token>

# wandb（不想用就跳過，並在 STEP 3 指令最前面加 USE_WANDB=0）
wandb login <YOUR_API_KEY>    # key: https://wandb.ai/authorize
```

---

## STEP 2 — Dry-run:用極小 step 驗證整條流程(~10 分鐘)

> 正式訓練前先跑這個,把整條 `訓練 → 存檔 → 自動上傳` 用 10 個 step 走一遍。
> 驗證:模型下載/載入、資料集+modality 對得上、**batch 64 不 OOM**、checkpoint 存檔、`&&` 自動上傳 + HF 寫入權限。
> 順便把 ~7GB base model 快取起來,正式訓練(STEP 3)就不用再下載。

```bash
cd ~/Isaac-GR00T
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

hf auth whoami                # 先確認 HF 已登入且有寫入權限

# 輸出放 repo 內的 checkpoints/_dryrun（不用 /tmp，避免重開機被清空）；上傳到私有 -dryrun 測試 repo。
# 用跟正式訓練「相同的 GLOBAL_BATCH_SIZE=128」→ 這裡不 OOM（實測 ~70GB/96GB），正式跑才安全。
USE_WANDB=0 CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
MAX_STEPS=10 SAVE_STEPS=5 SAVE_TOTAL_LIMIT=2 GLOBAL_BATCH_SIZE=128 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/_dryrun \
&& uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-dryrun \
     ~/Isaac-GR00T/checkpoints/_dryrun . --repo-type model --private --exclude '*optimizer.pt'
```

**通過標準(全部要出現):**

- 訓練沒 OOM,跑完 10 步,終端出現 `Model saved to ~/Isaac-GR00T/checkpoints/_dryrun`
- `~/Isaac-GR00T/checkpoints/_dryrun` 底下有 `checkpoint-5`、`checkpoint-10`
- `&&` 之後的 `hf upload` 成功(沒有 401/403 權限錯),HF 上出現私有的 `...-dryrun` repo

**(可選)順便驗證 STEP 4 的 eval 指令能跑:**

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path ~/Isaac-GR00T/checkpoints/_dryrun/checkpoint-10 \
  --traj-ids 0 --execution-horizon 16 --steps 50
# 註：dry-run 只練 10 步，MSE 會很大很正常，這裡只驗「指令能跑通」，不是看數值。
```

**清理 dry-run 產物:**

```bash
rm -rf ~/Isaac-GR00T/checkpoints/_dryrun
# 私有測試 repo 想刪的話，到 HF 網頁 Settings 刪 `...-dryrun`（或用當前 hf CLI 支援的刪除指令）。
```

通過後再進 STEP 3 正式訓練。

---

## STEP 3 — 訓練 + 訓練成功自動上傳

> **務必在 tmux 裡跑**,否則 SSH 一斷,~10–13 小時的訓練就死了。

```bash
tmux new -s train             # 斷線後用 `tmux attach -t train` 重新接上

cd ~/Isaac-GR00T
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
MAX_STEPS=20000 SAVE_STEPS=5000 SAVE_TOTAL_LIMIT=4 GLOBAL_BATCH_SIZE=128 \
DATALOADER_NUM_WORKERS=12 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune \
  --wandb-project so101-bimanual \
  --experiment-name pickvials-n1p7-run1 \
&& uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
     ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune . \
     --repo-type model --exclude '*optimizer.pt'
```

- 離開 tmux(訓練繼續):`Ctrl-b` 然後按 `d`。重新接上:`tmux attach -t train`。
- `&&` = 只有訓練成功(exit 0)才上傳;訓練失敗不會亂傳。
- `--exclude '*optimizer.pt'` = 略過只有續訓才用得到的 optimizer(省一半上傳量)。想要可續訓的完整備份就拿掉它。

---

## STEP 4 — 訓練後:健康檢查 + 選 checkpoint

```bash
cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

# 健康檢查：MSE 應隨 step「單調下降」。掃幾個看趨勢即可（不是挑最低點）
for ckpt in 5000 10000 15000 20000; do
  echo "=== checkpoint-$ckpt ==="
  uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune/checkpoint-$ckpt \
    --traj-ids 0 --execution-horizon 16 --steps 400
done
```

**選擇定案**:官方預設用終點 `checkpoint-20000`,最終以 **實機 rollout 的 task success rate** 拍板。
open-loop MSE 只是「訓練有沒有在學」的健康檢查(`rising` = 訓練壞掉,不是過擬合),**不是** checkpoint 選擇器。

---

## STEP 5 — 只在 STEP 3 的自動上傳失敗時,手動重傳

```bash
cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"
ls ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune          # 先確認有哪些 checkpoint
uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
  ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune . \
  --repo-type model --exclude '*optimizer.pt'
```

---

## 速查表

| 項目 | 值 / 說明 |
|---|---|
| 訓練時間(估) | ~10–13 小時(1× RTX PRO 6000, batch 128, 20K steps ≈ 14.5 epochs) |
| Checkpoint | 4 個(5K/10K/15K/20K),~40GB 本地 / ~28GB 上傳(排除 optimizer) |
| 磁碟 | 750GB 綽綽有餘(checkpoint ~40GB + venv/模型/資料集開銷 ~40GB) |
| 斷線保護 | **務必在 tmux 裡跑** |
| Dataloader workers | `DATALOADER_NUM_WORKERS=12`(16 核機器:留 4 核給主程序;每 worker ~5GB RAM,136GB 綽綽有餘) |
| 想改回可續訓的完整備份 | STEP 3/5 拿掉 `--exclude '*optimizer.pt'` |
| 關閉 wandb | STEP 3 指令最前面加 `USE_WANDB=0`,並移除 `--wandb-project` |
| VRAM 用量 | batch 128 實測 ~70GB / 96GB(26GB 餘裕);改 batch 務必同步調 `MAX_STEPS` 維持 epochs |
| 想練更久 | 加大 `MAX_STEPS`;中途斷線要接回同一數值則加 `--resume-from-checkpoint` |

## 超參數為什麼是這組(對應本資料集:159 episodes / 單任務)

| 參數 | 值 | 理由 |
|---|---|---|
| `MAX_STEPS` | 20000 | ≈14.5 epochs(= steps×batch/176700)。15K≈11 是甜蜜點,多跑當保險,靠 eval 挑最佳 |
| `GLOBAL_BATCH_SIZE` | 128 | 實測 ~70GB/96GB;官方 batch 32–640 都用同一 LR。改 batch 要同步調 `MAX_STEPS` 維持 epochs |
| `learning_rate` | 1e-4(預設) | 官方全系列驗證過,不隨 batch 改 |
| `grad_accum` | 1(預設) | 顯存夠,不需要;用了反而減少 weight update 次數 |
| `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` | 5000 / 4 | 4 個 checkpoint(5K/10K/15K/20K),全留供 eval 挑最佳 |
| `DATALOADER_NUM_WORKERS` | 12 | 16 核留 4 核給主程序;減少 shard 串流停頓 |
| `tune_visual` / `tune_llm` | False(預設) | 單任務 159 軌跡,解凍 backbone 會過擬合 |
